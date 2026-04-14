#![no_std]
use alloc::string::ToString;
use core::default::Default;
extern crate alloc;
use std::ffi::OsString;
use std::io;
use std::path::{Component, Path, PathBuf};

pub fn change_directory(current: &Path, dir: Option<&str>) -> io::Result<PathBuf> {
    let target = match dir {
        Some(path) => normalize_path(current, Path::new(path)),
        None => home_dir(),
    };

    match std::fs::metadata(&target) {
        Ok(meta) if meta.is_dir() => Ok(target),
        Ok(_) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            alloc::format!("not a directory: {}", target.display()),
        )),
        Err(err) => Err(err),
    }
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

fn normalize_path(base: &Path, input: &Path) -> PathBuf {
    let mut out = if input.is_absolute() {
        PathBuf::from("/")
    } else {
        base.to_path_buf()
    };

    for component in input.components() {
        match component {
            Component::RootDir => out = PathBuf::from("/"),
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            Component::Normal(seg) => out.push(seg),
            Component::Prefix(prefix) => out.push(OsString::from(prefix.as_os_str())),
        }
    }

    if out.as_os_str().is_empty() {
        PathBuf::from("/")
    } else {
        out
    }
}
