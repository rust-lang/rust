use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub(crate) fn get_rustc_version() -> String {
    let version_info =
        Command::new("rustc").stderr(Stdio::inherit()).args(&["-V"]).output().unwrap().stdout;
    String::from_utf8(version_info).unwrap()
}

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

pub(crate) fn get_rustc_path() -> PathBuf {
    let rustc_path = Command::new("rustup")
        .stderr(Stdio::inherit())
        .args(&["which", "rustc"])
        .output()
        .unwrap()
        .stdout;
    Path::new(String::from_utf8(rustc_path).unwrap().trim()).to_owned()
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

pub(crate) fn get_file_name(crate_name: &str, crate_type: &str) -> String {
    let file_name = Command::new("rustc")
        .stderr(Stdio::inherit())
        .args(&[
            "--crate-name",
            crate_name,
            "--crate-type",
            crate_type,
            "--print",
            "file-names",
            "-",
        ])
        .output()
        .unwrap()
        .stdout;
    let file_name = String::from_utf8(file_name).unwrap().trim().to_owned();
    assert!(!file_name.contains('\n'));
    assert!(file_name.contains(crate_name));
    file_name
}

/// Similar to `get_file_name`, but converts any dashes (`-`) in the `crate_name` to
/// underscores (`_`). This is specially made for the the rustc and cargo wrappers
/// which have a dash in the name, and that is not allowed in a crate name.
pub(crate) fn get_wrapper_file_name(crate_name: &str, crate_type: &str) -> String {
    let crate_name = crate_name.replace('-', "_");
    let wrapper_name = get_file_name(&crate_name, crate_type);
    wrapper_name.replace('_', "-")
}
