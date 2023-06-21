use anyhow::{format_err, Context, Result};
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

// Needed to set the script mode to executable.
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
// FIXME: what about Windows? Are default ACLs executable?

#[cfg(unix)]
use std::os::unix::fs::symlink as symlink_file;
#[cfg(windows)]
use std::os::windows::fs::symlink_file;

/// Converts a `&Path` to a UTF-8 `&str`.
pub fn path_to_str(path: &Path) -> Result<&str> {
    path.to_str()
        .ok_or_else(|| format_err!("path is not valid UTF-8 '{}'", path.display()))
}

/// Wraps `fs::copy` with a nicer error message.
pub fn copy<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> Result<u64> {
    if fs::symlink_metadata(&from)?.file_type().is_symlink() {
        let link = fs::read_link(&from)?;
        symlink_file(link, &to)?;
        Ok(0)
    } else {
        let amt = fs::copy(&from, &to).with_context(|| {
            format!(
                "failed to copy '{}' to '{}'",
                from.as_ref().display(),
                to.as_ref().display()
            )
        })?;
        Ok(amt)
    }
}

/// Wraps `fs::create_dir` with a nicer error message.
pub fn create_dir<P: AsRef<Path>>(path: P) -> Result<()> {
    fs::create_dir(&path)
        .with_context(|| format!("failed to create dir '{}'", path.as_ref().display()))?;
    Ok(())
}

/// Wraps `fs::create_dir_all` with a nicer error message.
pub fn create_dir_all<P: AsRef<Path>>(path: P) -> Result<()> {
    fs::create_dir_all(&path)
        .with_context(|| format!("failed to create dir '{}'", path.as_ref().display()))?;
    Ok(())
}

/// Wraps `fs::OpenOptions::create_new().open()` as executable, with a nicer error message.
pub fn create_new_executable<P: AsRef<Path>>(path: P) -> Result<fs::File> {
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    options.mode(0o755);
    let file = options
        .open(&path)
        .with_context(|| format!("failed to create file '{}'", path.as_ref().display()))?;
    Ok(file)
}

/// Wraps `fs::OpenOptions::create_new().open()`, with a nicer error message.
pub fn create_new_file<P: AsRef<Path>>(path: P) -> Result<fs::File> {
    let file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .with_context(|| format!("failed to create file '{}'", path.as_ref().display()))?;
    Ok(file)
}

/// Wraps `fs::File::open()` with a nicer error message.
pub fn open_file<P: AsRef<Path>>(path: P) -> Result<fs::File> {
    let file = fs::File::open(&path)
        .with_context(|| format!("failed to open file '{}'", path.as_ref().display()))?;
    Ok(file)
}

/// Wraps `remove_dir_all` with a nicer error message.
pub fn remove_dir_all<P: AsRef<Path>>(path: P) -> Result<()> {
    fs::remove_dir_all(path.as_ref())
        .with_context(|| format!("failed to remove dir '{}'", path.as_ref().display()))?;
    Ok(())
}

/// Wrap `fs::remove_file` with a nicer error message
pub fn remove_file<P: AsRef<Path>>(path: P) -> Result<()> {
    fs::remove_file(path.as_ref())
        .with_context(|| format!("failed to remove file '{}'", path.as_ref().display()))?;
    Ok(())
}

/// Copies the `src` directory recursively to `dst`. Both are assumed to exist
/// when this function is called.
pub fn copy_recursive(src: &Path, dst: &Path) -> Result<()> {
    copy_with_callback(src, dst, |_, _| Ok(()))
}

/// Copies the `src` directory recursively to `dst`. Both are assumed to exist
/// when this function is called. Invokes a callback for each path visited.
pub fn copy_with_callback<F>(src: &Path, dst: &Path, mut callback: F) -> Result<()>
where
    F: FnMut(&Path, fs::FileType) -> Result<()>,
{
    for entry in WalkDir::new(src).min_depth(1) {
        let entry = entry?;
        let file_type = entry.file_type();
        let path = entry.path().strip_prefix(src)?;
        let dst = dst.join(path);

        if file_type.is_dir() {
            create_dir(&dst)?;
        } else {
            copy(entry.path(), dst)?;
        }
        callback(path, file_type)?;
    }
    Ok(())
}

macro_rules! actor_field_default {
    () => { Default::default() };
    (= $expr:expr) => { $expr.into() }
}

/// Creates an "actor" with default values, setters for all fields, and Clap parser support.
macro_rules! actor {
    ($( #[ $attr:meta ] )+ pub struct $name:ident {
        $( $( #[ $field_attr:meta ] )+ $field:ident : $type:ty $(= $default:tt)*, )*
    }) => {
        $( #[ $attr ] )+
        #[derive(clap::Args)]
        pub struct $name {
            $( $( #[ $field_attr ] )+ #[arg(long, $(default_value = $default)*)] $field : $type, )*
        }

        impl Default for $name {
            fn default() -> $name {
                $name {
                    $($field : actor_field_default!($(= $default)*), )*
                }
            }
        }

        impl $name {
            $(pub fn $field(&mut self, value: $type) -> &mut Self {
                self.$field = value;
                self
            })*
        }
    }
}
