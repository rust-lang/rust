use std::path::{Path, PathBuf};

use xshell::{Shell, cmd};

pub(crate) fn list_rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut res = list_files(dir);
    res.retain(|it| {
        it.file_name().unwrap_or_default().to_str().unwrap_or_default().ends_with(".rs")
    });
    res
}

pub(crate) fn list_files(dir: &Path) -> Vec<PathBuf> {
    let mut res = Vec::new();
    let mut work = vec![dir.to_path_buf()];
    while let Some(dir) = work.pop() {
        for entry in dir.read_dir().unwrap() {
            let entry = entry.unwrap();
            let file_type = entry.file_type().unwrap();
            let path = entry.path();
            let is_hidden =
                path.file_name().unwrap_or_default().to_str().unwrap_or_default().starts_with('.');
            if !is_hidden {
                if file_type.is_dir() {
                    work.push(path);
                } else if file_type.is_file() {
                    res.push(path);
                }
            }
        }
    }
    res
}

pub(crate) fn detect_target(sh: &Shell) -> String {
    match std::env::var("RA_TARGET") {
        Ok(target) => target,
        _ => match cmd!(sh, "rustc --print=host-tuple").read() {
            Ok(target) => target,
            Err(e) => panic!("Failed to detect target: {e}\nPlease set RA_TARGET explicitly"),
        },
    }
}
