use camino::Utf8PathBuf;

#[cfg(target_family = "unix")]
mod linux;
#[cfg(target_family = "windows")]
mod windows;

pub trait Environment {
    fn host_triple(&self) -> String {
        std::env::var("PGO_HOST").expect("PGO_HOST environment variable missing")
    }

    fn python_binary(&self) -> &'static str;

    /// The rustc checkout, where the compiler source is located.
    fn checkout_path(&self) -> Utf8PathBuf;

    /// Path to the downloaded host LLVM.
    fn downloaded_llvm_dir(&self) -> Utf8PathBuf;

    /// Directory where the optimization artifacts (PGO/BOLT profiles, etc.)
    /// will be stored.
    fn opt_artifacts(&self) -> Utf8PathBuf;

    /// The main directory where the build occurs.
    fn build_root(&self) -> Utf8PathBuf;

    fn build_artifacts(&self) -> Utf8PathBuf {
        self.build_root().join("build").join(self.host_triple())
    }

    fn cargo_stage_0(&self) -> Utf8PathBuf {
        self.build_artifacts()
            .join("stage0")
            .join("bin")
            .join(format!("cargo{}", self.executable_extension()))
    }

    fn rustc_stage_0(&self) -> Utf8PathBuf {
        self.build_artifacts()
            .join("stage0")
            .join("bin")
            .join(format!("rustc{}", self.executable_extension()))
    }

    fn rustc_stage_2(&self) -> Utf8PathBuf {
        self.build_artifacts()
            .join("stage2")
            .join("bin")
            .join(format!("rustc{}", self.executable_extension()))
    }

    /// Path to the built rustc-perf benchmark suite.
    fn rustc_perf_dir(&self) -> Utf8PathBuf {
        self.opt_artifacts().join("rustc-perf")
    }

    /// Download and/or compile rustc-perf.
    fn prepare_rustc_perf(&self) -> anyhow::Result<()>;

    fn supports_bolt(&self) -> bool;

    /// What is the extension of binary executables in this environment?
    fn executable_extension(&self) -> &'static str;

    /// List of test paths that should be skipped when testing the optimized artifacts.
    fn skipped_tests(&self) -> &'static [&'static str];
}

pub fn create_environment() -> Box<dyn Environment> {
    #[cfg(target_family = "unix")]
    return Box::new(linux::LinuxEnvironment);
    #[cfg(target_family = "windows")]
    return Box::new(windows::WindowsEnvironment::new());
}
