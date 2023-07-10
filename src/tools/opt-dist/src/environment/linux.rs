use crate::environment::Environment;
use crate::exec::cmd;
use crate::utils::io::copy_directory;
use camino::{Utf8Path, Utf8PathBuf};

pub(super) struct LinuxEnvironment;

impl Environment for LinuxEnvironment {
    fn python_binary(&self) -> &'static str {
        "python3"
    }

    fn checkout_path(&self) -> Utf8PathBuf {
        Utf8PathBuf::from("/checkout")
    }

    fn downloaded_llvm_dir(&self) -> Utf8PathBuf {
        Utf8PathBuf::from("/rustroot")
    }

    fn opt_artifacts(&self) -> Utf8PathBuf {
        Utf8PathBuf::from("/tmp/tmp-multistage/opt-artifacts")
    }

    fn build_root(&self) -> Utf8PathBuf {
        self.checkout_path().join("obj")
    }

    fn prepare_rustc_perf(&self) -> anyhow::Result<()> {
        // /tmp/rustc-perf comes from the x64 dist Dockerfile
        copy_directory(Utf8Path::new("/tmp/rustc-perf"), &self.rustc_perf_dir())?;
        cmd(&[self.cargo_stage_0().as_str(), "build", "-p", "collector"])
            .workdir(&self.rustc_perf_dir())
            .env("RUSTC", &self.rustc_stage_0().into_string())
            .env("RUSTC_BOOTSTRAP", "1")
            .run()?;
        Ok(())
    }

    fn supports_bolt(&self) -> bool {
        true
    }

    fn executable_extension(&self) -> &'static str {
        ""
    }

    fn skipped_tests(&self) -> &'static [&'static str] {
        &[
            // Fails because of linker errors, as of June 2023.
            "tests/ui/process/nofile-limit.rs",
        ]
    }
}
