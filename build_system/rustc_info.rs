use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub(crate) fn get_host_triple() -> String {
    let version_info =
        Command::new("rustc").stderr(Stdio::inherit()).args(&["-vV"]).output().unwrap().stdout;
    String::from_utf8(version_info)
        .unwrap()
        .lines()
        .to_owned()
        .find(|line| line.starts_with("host"))
        .unwrap()
        .split(":")
        .nth(1)
        .unwrap()
        .trim()
        .to_owned()
}

pub(crate) fn get_default_sysroot() -> PathBuf {
    let default_sysroot = Command::new("rustc")
        .stderr(Stdio::inherit())
        .args(&["--print", "sysroot"])
        .output()
        .unwrap()
        .stdout;
    Path::new(String::from_utf8(default_sysroot).unwrap().trim()).to_owned()
}

pub(crate) fn get_dylib_name(crate_name: &str) -> String {
    let dylib_name = Command::new("rustc")
        .stderr(Stdio::inherit())
        .args(&["--crate-name", crate_name, "--crate-type", "dylib", "--print", "file-names", "-"])
        .output()
        .unwrap()
        .stdout;
    let dylib_name = String::from_utf8(dylib_name).unwrap().trim().to_owned();
    assert!(!dylib_name.contains('\n'));
    assert!(dylib_name.contains(crate_name));
    dylib_name
}
