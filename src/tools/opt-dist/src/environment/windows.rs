use crate::environment::Environment;
use crate::exec::cmd;
use crate::utils::io::move_directory;
use camino::Utf8PathBuf;
use std::io::Cursor;
use zip::ZipArchive;

pub(super) struct WindowsEnvironment {
    checkout_dir: Utf8PathBuf,
}

impl WindowsEnvironment {
    pub fn new() -> Self {
        Self { checkout_dir: std::env::current_dir().unwrap().try_into().unwrap() }
    }
}

impl Environment for WindowsEnvironment {
    fn python_binary(&self) -> &'static str {
        "python"
    }

    fn checkout_path(&self) -> Utf8PathBuf {
        self.checkout_dir.clone()
    }

    fn downloaded_llvm_dir(&self) -> Utf8PathBuf {
        self.checkout_path().join("citools").join("clang-rust")
    }

    fn opt_artifacts(&self) -> Utf8PathBuf {
        self.checkout_path().join("opt-artifacts")
    }

    fn build_root(&self) -> Utf8PathBuf {
        self.checkout_path()
    }

    fn prepare_rustc_perf(&self) -> anyhow::Result<()> {
        // FIXME: add some mechanism for synchronization of this commit SHA with
        // Linux (which builds rustc-perf in a Dockerfile)
        // rustc-perf version from 2023-05-30
        const PERF_COMMIT: &str = "8b2ac3042e1ff2c0074455a0a3618adef97156b1";

        let url = format!("https://github.com/rust-lang/rustc-perf/archive/{PERF_COMMIT}.zip");
        let response = reqwest::blocking::get(url)?.error_for_status()?.bytes()?.to_vec();

        let mut archive = ZipArchive::new(Cursor::new(response))?;
        archive.extract(self.rustc_perf_dir())?;
        move_directory(
            &self.rustc_perf_dir().join(format!("rustc-perf-{PERF_COMMIT}")),
            &self.rustc_perf_dir(),
        )?;

        cmd(&[self.cargo_stage_0().as_str(), "build", "-p", "collector"])
            .workdir(&self.rustc_perf_dir())
            .env("RUSTC", &self.rustc_stage_0().into_string())
            .env("RUSTC_BOOTSTRAP", "1")
            .run()?;

        Ok(())
    }

    fn supports_bolt(&self) -> bool {
        false
    }

    fn executable_extension(&self) -> &'static str {
        ".exe"
    }

    fn skipped_tests(&self) -> &'static [&'static str] {
        &[
            // Fails as of June 2023.
            "tests\\codegen\\vec-shrink-panik.rs",
        ]
    }
}
