use anyhow::Context;
use camino::Utf8Path;
use fs_extra::dir::CopyOptions;
use std::fs::File;

/// Delete and re-create the directory.
pub fn reset_directory(path: &Utf8Path) -> anyhow::Result<()> {
    log::info!("Resetting directory {path}");
    let _ = std::fs::remove_dir(path);
    std::fs::create_dir_all(path)?;
    Ok(())
}

pub fn copy_directory(src: &Utf8Path, dst: &Utf8Path) -> anyhow::Result<()> {
    log::info!("Copying directory {src} to {dst}");
    fs_extra::dir::copy(src, dst, &CopyOptions::default().copy_inside(true))?;
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
