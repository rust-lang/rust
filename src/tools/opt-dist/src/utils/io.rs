use std::fs::File;
use std::path::Path;

use anyhow::Context;
use camino::{Utf8Path, Utf8PathBuf};
use fs_extra::dir::CopyOptions;

/// Delete and re-create the directory.
pub fn reset_directory(path: &Utf8Path) -> anyhow::Result<()> {
    log::info!("Resetting directory {path}");
    let _ = std::fs::remove_dir_all(path);
    std::fs::create_dir_all(path)?;
    Ok(())
}

pub fn copy_directory(src: &Utf8Path, dst: &Utf8Path) -> anyhow::Result<()> {
    log::info!("Copying directory {src} to {dst}");
    fs_extra::dir::copy(src, dst, &CopyOptions::default().copy_inside(true))?;
    Ok(())
}

pub fn copy_file<S: AsRef<Path>, D: AsRef<Path>>(src: S, dst: D) -> anyhow::Result<()> {
    log::info!("Copying file {} to {}", src.as_ref().display(), dst.as_ref().display());
    std::fs::copy(src.as_ref(), dst.as_ref())?;
    Ok(())
}

#[allow(unused)]
pub fn move_directory(src: &Utf8Path, dst: &Utf8Path) -> anyhow::Result<()> {
    log::info!("Moving directory {src} to {dst}");
    fs_extra::dir::move_dir(src, dst, &CopyOptions::default().content_only(true))?;
    Ok(())
}

/// Counts all children of a directory (non-recursively).
pub fn count_files(dir: &Utf8Path) -> anyhow::Result<u64> {
    Ok(std::fs::read_dir(dir)?.count() as u64)
}

pub fn delete_directory(path: &Utf8Path) -> anyhow::Result<()> {
    log::info!("Deleting directory `{path}`");
    std::fs::remove_dir_all(path.as_std_path())
        .context(format!("Cannot remove directory {path}"))?;
    Ok(())
}

pub fn unpack_archive(path: &Utf8Path, dest_dir: &Utf8Path) -> anyhow::Result<()> {
    log::info!("Unpacking directory `{path}` into `{dest_dir}`");

    assert!(path.as_str().ends_with(".tar.xz"));
    let file = File::open(path.as_std_path())?;
    let file = xz::read::XzDecoder::new(file);
    let mut archive = tar::Archive::new(file);
    archive.unpack(dest_dir.as_std_path())?;
    Ok(())
}

/// Returns paths in the given `dir` (non-recursively), optionally with the given `suffix`.
/// The `suffix` should contain the leading dot.
pub fn get_files_from_dir(
    dir: &Utf8Path,
    suffix: Option<&str>,
) -> anyhow::Result<Vec<Utf8PathBuf>> {
    let path = format!("{dir}/*{}", suffix.unwrap_or(""));

    Ok(glob::glob(&path)?
        .map(|p| p.map(|p| Utf8PathBuf::from_path_buf(p).unwrap()))
        .collect::<Result<Vec<_>, _>>()?)
}

/// Finds a single file in the specified `directory` with the given `prefix` and `suffix`.
pub fn find_file_in_dir(
    directory: &Utf8Path,
    prefix: &str,
    suffix: &str,
) -> anyhow::Result<Utf8PathBuf> {
    let files =
        glob::glob(&format!("{directory}/{prefix}*{suffix}"))?.collect::<Result<Vec<_>, _>>()?;
    match files.len() {
        0 => Err(anyhow::anyhow!("No file with prefix {prefix} found in {directory}")),
        1 => Ok(Utf8PathBuf::from_path_buf(files[0].clone()).unwrap()),
        _ => Err(anyhow::anyhow!(
            "More than one file with prefix {prefix} found in {directory}: {:?}",
            files
        )),
    }
}
