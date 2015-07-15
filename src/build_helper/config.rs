use std::env;
use std::path::{Path, PathBuf};
use std::fs::PathExt;

/// Collect the environment variables passed to the build script.
/// Note: To determine the root directory of the rust source repo we simply
/// concat "../.." to the manifest directory of the Cargo package being built.
/// This will only work if all Cargo packages are placed under `src/`.
pub struct Config {
    manifest_dir : PathBuf,
    out_dir : PathBuf,
    llvm_root : PathBuf,
    target : String,
    host : String,
    njobs : u8,
    profile : String
}

impl Config {
    pub fn new() -> Config {
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")
                                         .expect("CARGO_MANIFEST_DIR"));
        let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
        let llvm_root = PathBuf::from(env::var("CFG_LLVM_ROOT")
                                      .expect("CFG_LLVM_ROOT"));
        let target = env::var("TARGET").expect("TARGET");
        let host = env::var("HOST").expect("HOST");
        let njobs : u8 = env::var("NUM_JOBS").expect("NUM_JOBS")
            .parse().expect("parse NUM_JOBS");
        let profile = env::var("PROFILE").expect("PROFILE");
        Config {
            manifest_dir : manifest_dir,
            out_dir : out_dir,
            llvm_root : llvm_root,
            target : target,
            host : host,
            njobs : njobs,
            profile : profile
        }
    }

    /// Root directory of the Rust project
    pub fn root_dir(&self) -> PathBuf {
        self.manifest_dir.join("..").join("..")
    }

    /// Parent directory of all Cargo packages
    pub fn src_dir(&self) -> PathBuf {
        self.manifest_dir.join("..")
    }

    /// Output directory of the Cargo package
    pub fn out_dir(&self) -> &Path {
        &self.out_dir
    }

    /// Build artifacts directory for LLVM
    pub fn llvm_build_artifacts_dir(&self) -> PathBuf {
        let dirs = vec![".", "Release", "Release+Asserts",
                        "Debug", "Debug+Asserts"];
        for d in &dirs {
            let artifacts_dir = self.llvm_root.join(d);
            if artifacts_dir.join("bin").join("llc").is_file() {
                return artifacts_dir;
            } else if artifacts_dir.join("bin").join("llc.exe").is_file() {
                return artifacts_dir;
            }
        }
        panic!("Directory supplied to CFG_LLVM_ROOT does not \
                contain valid LLVM build.");
    }

    /// Target triple being compiled for
    pub fn target(&self) -> &str {
        &self.target
    }

    /// Host triple of the rustc compiler
    pub fn host(&self) -> &str {
        &self.host
    }

    /// Number of parallel jobs to run
    pub fn njobs(&self) -> u8 {
        self.njobs
    }

    /// Profile being built
    pub fn profile(&self) -> &str {
        &self.profile
    }
}
