package com.ernesto.pictureupgrader

import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.provider.Settings
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.SeekBar
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import com.ernesto.pictureupgrader.databinding.ActivityMainBinding
import com.ernesto.pictureupgrader.databinding.DialogResolutionReductionBinding
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    companion object {
        private const val GALLERY_REQ = 0x3a
        private const val REQUEST_CODE = 0x3f

        init {
            System.loadLibrary("pictureupgrader")
        }
    }

    private lateinit var binding: ActivityMainBinding
    private var dlg : AlertDialog? = null
    private var selectedImageUri: Uri? = null
    private var selectedImagePath: String? = null

    private external fun imgSupResolution(inPath: String, outPath: String, modelDir: String) : Boolean
    private external fun imgColouration(inPath: String, outPath: String, modelDir: String) : Boolean

    private fun checkTargetExists(context: Context, subDir: String) : Boolean {
        val dir = context.getExternalFilesDir(null)
        val subDirFile = File(dir, subDir)
        return subDirFile.exists()
    }

    private fun copyAssetsSubDirToDir(context: Context, subDir: String): File {
        val assetManager = context.assets
        val dir = context.getExternalFilesDir(null)
        val subDirFile = File(dir, subDir)
        if (!subDirFile.exists()) {
            subDirFile.mkdirs()
        }
        val assets = assetManager.list(subDir)
        assets?.forEach { asset ->
            val assetFile = File(subDirFile, asset)
            assetManager.open("$subDir/$asset").use { inputStream ->
                FileOutputStream(assetFile).use { outputStream ->
                    inputStream.copyTo(outputStream)
                    outputStream.flush()
                }
            }
        }
        return subDirFile
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        if (Build.VERSION.SDK_INT > Build.VERSION_CODES.S) {
            if(!Environment.isExternalStorageManager()) {
                val intent = Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION)
                startActivityForResult(intent, REQUEST_CODE)
            }
        } else {
            if(ActivityCompat.checkSelfPermission(this, android.Manifest.permission.
                WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED)
                requestPermissions(arrayOf(android.Manifest.permission.WRITE_EXTERNAL_STORAGE), REQUEST_CODE)
        }

        binding.btnSelect.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            startActivityForResult(intent, GALLERY_REQ)
        }

        binding.btnSuperResolution.setOnClickListener {
            selectedImagePath?.let { imgPath ->
                val bitmap = BitmapFactory.decodeFile(imgPath)
                if(bitmap.height > 1500 || bitmap.width > 1500) {
                    AlertDialog.Builder(this).apply {
                        setTitle(R.string.warn_title)
                        setMessage(R.string.large_warn)
                        setCancelable(false)
                        setPositiveButton("Yes") { _,_ ->
                            processImage(::imgSupResolution, "Super Resolution")
                        }
                        setNegativeButton("No", null)
                    }.create().show()
                } else processImage(::imgSupResolution, "Super Resolution")
            } ?: run {
                Toast.makeText(this, "Please select an image first.", Toast.LENGTH_SHORT).show()
            }
        }

        binding.btnColorization.setOnClickListener {
            processImage(::imgColouration, "Colouration")
        }

        binding.btnDownsmp.setOnClickListener {
            selectedImagePath?.let { imagePath ->
                val drrb = DialogResolutionReductionBinding.inflate(layoutInflater)
                val bitmap = BitmapFactory.decodeFile(imagePath)

                with(drrb.seekBar) {
                    max = 100
                    progress = 80
                    setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                        override fun onProgressChanged(seekBar: SeekBar, progress: Int, fromUser: Boolean) {
                            drrb.currentValueTextView.text = progress.toString()
                        }
                        override fun onStartTrackingTouch(seekBar: SeekBar) {}
                        override fun onStopTrackingTouch(seekBar: SeekBar) {}
                    })
                    AlertDialog.Builder(this@MainActivity).apply {
                        setTitle(R.string.down_sampling)
                        setMessage("${getString(R.string.current_res)}: ${bitmap.width} x ${bitmap.height}")
                        setView(drrb.root)
                        setPositiveButton("OK") { _, _ ->
                            val file = selectedImagePath?.let { it1 -> File(it1) }
                            thread {
                                try {
                                    val scaledBitmap = scaleBitmap(bitmap, progress.toFloat() / 100.0f)
                                    FileOutputStream(file).use { out ->
                                        scaledBitmap?.compress(Bitmap.CompressFormat.JPEG, 95, out)
                                    }
                                    runOnUiThread {
                                        binding.imageView.setImageBitmap(BitmapFactory.decodeFile(selectedImagePath))
                                        Toast.makeText(this@MainActivity, "Down sampling completed!", Toast.LENGTH_SHORT).show()
                                    }
                                    selectedImagePath = null
                                } catch (e: IOException) {
                                    e.printStackTrace()
                                    runOnUiThread {
                                        Toast.makeText(this@MainActivity, "Down sampling failed!", Toast.LENGTH_SHORT).show()
                                    }
                                }
                            }
                        }
                        setNegativeButton("Cancel", null)
                    }.create().show()
                }
            } ?: run {
                Toast.makeText(this, "Please select an image first.", Toast.LENGTH_SHORT).show()
            }
        }

        if(!checkTargetExists(this, "models")) {
            dlg = AlertDialog.Builder(this).apply {
                setTitle(R.string.in_progress)
                setMessage(R.string.extracting)
                setCancelable(false)
            }.create()
            dlg?.show()
            thread {
                copyAssetsSubDirToDir(this, "models")
                runOnUiThread {
                    dlg?.dismiss()
                }
            }
        }
    }

    private fun scaleBitmap(bitmap: Bitmap, scale: Float): Bitmap? {
        val matrix = android.graphics.Matrix()
        matrix.postScale(scale, scale)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE) {
            if (grantResults.isEmpty() || grantResults[0] != PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Insufficient permission!", Toast.LENGTH_LONG).show()
                finish()
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == GALLERY_REQ && resultCode == RESULT_OK && data != null) {
            selectedImageUri = data.data
            selectedImagePath = getPathFromUri(selectedImageUri!!)
            val bitmap = BitmapFactory.decodeFile(selectedImagePath)
            binding.imageView.setImageBitmap(bitmap)
        }
    }

    private fun processImagePath(imagePath: String): String {
        val file = File(imagePath)
        val parentDir = file.parent
        val fileName = file.nameWithoutExtension
        val processedFileName = "${fileName}_processed.jpg"

        return if (parentDir.isNullOrEmpty()) {
            processedFileName
        } else {
            "$parentDir/$processedFileName"
        }
    }

    private fun processImage(processFunc: (String, String, String) -> Boolean, processName: String) {
        selectedImagePath?.let { imagePath ->
            val progressBar = ProgressBar(this)
            progressBar.layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
            dlg = AlertDialog.Builder(this).apply {
                setTitle(R.string.in_progress)
                setView(progressBar)
                setMessage(getString(R.string.content_in_progress).format(processName))
                setCancelable(false)
            }.create()
            dlg?.show()
            thread {
                val outPath = processImagePath(imagePath)
                val modelDir = File(getExternalFilesDir(null), "models").absolutePath
                val success = processFunc(imagePath, outPath, modelDir)
                runOnUiThread {
                    dlg?.dismiss()
                    if (success) {
                        val bitmap = BitmapFactory.decodeFile(outPath)
                        binding.imageView.setImageBitmap(bitmap)
                        Toast.makeText(this, "$processName completed.", Toast.LENGTH_SHORT).show()
                    } else {
                        Toast.makeText(this, "$processName failed.", Toast.LENGTH_SHORT).show()
                    }
                }
                selectedImagePath = null
            }
        } ?: run {
            Toast.makeText(this, "Please select an image first.", Toast.LENGTH_SHORT).show()
        }
    }

    private fun getPathFromUri(uri: Uri): String {
        return contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            cursor.moveToNext()
            cursor.getString(cursor.getColumnIndexOrThrow("_data"))
        } ?: throw IllegalArgumentException("Cannot get path from URI")
    }
}