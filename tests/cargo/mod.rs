use cargo_metadata::{Message::CompilerArtifact, MetadataCommand};
use std::env;
use std::ffi::OsStr;
use std::mem;
use std::path::PathBuf;
use std::process::Command;

pub struct BuildInfo {
    cargo_target_dir: PathBuf,
}

impl BuildInfo {
    pub fn new() -> Self {
        let data = MetadataCommand::new().exec().unwrap();
        Self {
            cargo_target_dir: data.target_directory,
        }
    }

    pub fn host_lib(&self) -> PathBuf {
        if let Some(path) = option_env!("HOST_LIBS") {
            PathBuf::from(path)
        } else {
            self.cargo_target_dir.join(env!("PROFILE"))
        }
    }

    pub fn target_lib(&self) -> PathBuf {
        if let Some(path) = option_env!("TARGET_LIBS") {
            path.into()
        } else {
            let mut dir = self.cargo_target_dir.clone();
            if let Some(target) = env::var_os("CARGO_BUILD_TARGET") {
                dir.push(target);
            }
            dir.push(env!("PROFILE"));
            dir
        }
    }

    pub fn clippy_driver_path(&self) -> PathBuf {
        if let Some(path) = option_env!("CLIPPY_DRIVER_PATH") {
            PathBuf::from(path)
        } else {
            self.target_lib().join("clippy-driver")
        }
    }

    // When we'll want to use `extern crate ..` for a dependency that is used
    // both by the crate and the compiler itself, we can't simply pass -L flags
    // as we'll get a duplicate matching versions. Instead, disambiguate with
    // `--extern dep=path`.
    // See https://github.com/rust-lang/rust-clippy/issues/4015.
    pub fn third_party_crates() -> Vec<(&'static str, PathBuf)> {
        const THIRD_PARTY_CRATES: [&str; 3] = ["serde", "regex", "clippy_lints"];
        let cargo = env::var_os("CARGO");
        let cargo = cargo.as_deref().unwrap_or_else(|| OsStr::new("cargo"));
        let output = Command::new(cargo)
            .arg("build")
            .arg("--test=compile-test")
            .arg("--message-format=json")
            .output()
            .unwrap();

        let mut crates = Vec::with_capacity(THIRD_PARTY_CRATES.len());
        for message in cargo_metadata::parse_messages(output.stdout.as_slice()) {
            if let CompilerArtifact(mut artifact) = message.unwrap() {
                if let Some(&krate) = THIRD_PARTY_CRATES.iter().find(|&&krate| krate == artifact.target.name) {
                    crates.push((krate, mem::take(&mut artifact.filenames[0])));
                }
            }
        }
        crates
    }
}
