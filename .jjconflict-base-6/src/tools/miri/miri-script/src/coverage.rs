use std::path::PathBuf;

use anyhow::{Context, Result};
use path_macro::path;
use tempfile::TempDir;
use xshell::cmd;

use crate::util::MiriEnv;

/// CoverageReport can generate code coverage reports for miri.
pub struct CoverageReport {
    /// path is a temporary directory where intermediate coverage artifacts will be stored.
    /// (The final output will be stored in a permanent location.)
    path: TempDir,
}

impl CoverageReport {
    /// Creates a new CoverageReport.
    ///
    /// # Errors
    ///
    /// An error will be returned if a temporary directory could not be created.
    pub fn new() -> Result<Self> {
        Ok(Self { path: TempDir::new()? })
    }

    /// add_env_vars will add the required environment variables to MiriEnv `e`.
    pub fn add_env_vars(&self, e: &mut MiriEnv) -> Result<()> {
        let mut rustflags = e.sh.var("RUSTFLAGS")?;
        rustflags.push_str(" -C instrument-coverage");
        e.sh.set_var("RUSTFLAGS", rustflags);

        // Copy-pasting from: https://doc.rust-lang.org/rustc/instrument-coverage.html#instrumentation-based-code-coverage
        // The format symbols below have the following meaning:
        // - %p - The process ID.
        // - %Nm - the instrumented binary’s signature:
        //   The runtime creates a pool of N raw profiles, used for on-line
        //   profile merging. The runtime takes care of selecting a raw profile
        //   from the pool, locking it, and updating it before the program
        //   exits. N must be between 1 and 9, and defaults to 1 if omitted
        //   (with simply %m).
        //
        // Additionally the default for LLVM_PROFILE_FILE is default_%m_%p.profraw.
        // So we just use the same template, replacing "default" with "miri".
        let file_template = self.path.path().join("miri_%m_%p.profraw");
        e.sh.set_var("LLVM_PROFILE_FILE", file_template);
        Ok(())
    }

    /// show_coverage_report will print coverage information using the artifact
    /// files in `self.path`.
    pub fn show_coverage_report(&self, e: &MiriEnv, features: &[String]) -> Result<()> {
        let profraw_files = self.profraw_files()?;

        let profdata_bin = path!(e.libdir / ".." / "bin" / "llvm-profdata");

        let merged_file = path!(e.miri_dir / "target" / "coverage.profdata");

        // Merge the profraw files
        cmd!(e.sh, "{profdata_bin} merge -sparse {profraw_files...} -o {merged_file}")
            .quiet()
            .run()?;

        // Create the coverage report.
        let cov_bin = path!(e.libdir / ".." / "bin" / "llvm-cov");
        let miri_bin = e
            .build_get_binary(".", features)
            .context("failed to get filename of miri executable")?;
        cmd!(
            e.sh,
            "{cov_bin} report --instr-profile={merged_file} --object {miri_bin} --sources src/"
        )
        .run()?;

        println!("Profile data saved in {}", merged_file.display());
        Ok(())
    }

    /// profraw_files returns the profraw files in `self.path`.
    ///
    /// # Errors
    ///
    /// An error will be returned if `self.path` can't be read.
    fn profraw_files(&self) -> Result<Vec<PathBuf>> {
        Ok(std::fs::read_dir(&self.path)?
            .filter_map(|r| r.ok())
            .filter(|e| e.file_type().is_ok_and(|t| t.is_file()))
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "profraw"))
            .collect())
    }
}
