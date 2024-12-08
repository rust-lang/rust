use std::ffi::{OsStr, OsString};
use std::io::BufRead;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::{env, iter, thread};

use anyhow::{Context, Result, anyhow, bail};
use dunce::canonicalize;
use path_macro::path;
use xshell::{Cmd, Shell, cmd};

pub fn miri_dir() -> std::io::Result<PathBuf> {
    const MIRI_SCRIPT_ROOT_DIR: &str = env!("CARGO_MANIFEST_DIR");
    Ok(canonicalize(MIRI_SCRIPT_ROOT_DIR)?.parent().unwrap().into())
}

/// Queries the active toolchain for the Miri dir.
pub fn active_toolchain() -> Result<String> {
    let sh = Shell::new()?;
    sh.change_dir(miri_dir()?);
    let stdout = cmd!(sh, "rustup show active-toolchain").read()?;
    Ok(stdout.split_whitespace().next().context("Could not obtain active Rust toolchain")?.into())
}

pub fn flagsplit(flags: &str) -> Vec<String> {
    // This code is taken from `RUSTFLAGS` handling in cargo.
    flags.split(' ').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect()
}

/// Some extra state we track for building Miri, such as the right RUSTFLAGS.
#[derive(Clone)]
pub struct MiriEnv {
    /// miri_dir is the root of the miri repository checkout we are working in.
    pub miri_dir: PathBuf,
    /// active_toolchain is passed as `+toolchain` argument to cargo/rustc invocations.
    toolchain: String,
    /// Extra flags to pass to cargo.
    cargo_extra_flags: Vec<String>,
    /// The rustc sysroot
    pub sysroot: PathBuf,
    /// The shell we use.
    pub sh: Shell,
    /// The library dir in the sysroot.
    pub libdir: PathBuf,
}

impl MiriEnv {
    pub fn new() -> Result<Self> {
        let toolchain = active_toolchain()?;
        let sh = Shell::new()?; // we are preserving the current_dir on this one, so paths resolve properly!
        let miri_dir = miri_dir()?;

        let sysroot = cmd!(sh, "rustc +{toolchain} --print sysroot").read()?.into();
        let target_output = cmd!(sh, "rustc +{toolchain} --version --verbose").read()?;
        let rustc_meta = rustc_version::version_meta_for(&target_output)?;
        let libdir = path!(sysroot / "lib" / "rustlib" / rustc_meta.host / "lib");

        // Determine some toolchain properties
        if !libdir.exists() {
            eprintln!("Something went wrong determining the library dir.");
            eprintln!("I got {} but that does not exist.", libdir.display());
            eprintln!("Please report a bug at https://github.com/rust-lang/miri/issues.");
            std::process::exit(2);
        }

        // Hard-code the target dir, since we rely on all binaries ending up in the same spot.
        sh.set_var("CARGO_TARGET_DIR", path!(miri_dir / "target"));

        // We configure dev builds to not be unusably slow.
        let devel_opt_level =
            std::env::var_os("CARGO_PROFILE_DEV_OPT_LEVEL").unwrap_or_else(|| "2".into());
        sh.set_var("CARGO_PROFILE_DEV_OPT_LEVEL", devel_opt_level);

        // Compute rustflags.
        let rustflags = {
            let mut flags = OsString::new();
            // We set the rpath so that Miri finds the private rustc libraries it needs.
            // (This only makes sense on Unix.)
            if cfg!(unix) {
                flags.push("-C link-args=-Wl,-rpath,");
                flags.push(&libdir);
            }
            // Enable rustc-specific lints (ignored without `-Zunstable-options`).
            flags.push(
                " -Zunstable-options -Wrustc::internal -Wrust_2018_idioms -Wunused_lifetimes",
            );
            // Add user-defined flags.
            if let Some(value) = std::env::var_os("RUSTFLAGS") {
                flags.push(" ");
                flags.push(value);
            }
            flags
        };
        sh.set_var("RUSTFLAGS", rustflags);

        // On Windows, the `-Wl,-rpath,` above does not help. Instead we add the libdir to the PATH,
        // so that Windows can find the DLLs.
        if cfg!(windows) {
            let old_path = sh.var("PATH")?;
            let new_path =
                env::join_paths(iter::once(libdir.clone()).chain(env::split_paths(&old_path)))?;
            sh.set_var("PATH", new_path);
        }

        // Get extra flags for cargo.
        let cargo_extra_flags = std::env::var("CARGO_EXTRA_FLAGS").unwrap_or_default();
        let mut cargo_extra_flags = flagsplit(&cargo_extra_flags);
        if cargo_extra_flags.iter().any(|a| a == "--release" || a.starts_with("--profile")) {
            // This makes binaries end up in different paths, let's not do that.
            eprintln!(
                "Passing `--release` or `--profile` in `CARGO_EXTRA_FLAGS` will totally confuse miri-script, please don't do that."
            );
            std::process::exit(1);
        }
        // Also set `-Zroot-dir` for cargo, to print diagnostics relative to the miri dir.
        cargo_extra_flags.push(format!("-Zroot-dir={}", miri_dir.display()));

        Ok(MiriEnv { miri_dir, toolchain, sh, sysroot, cargo_extra_flags, libdir })
    }

    pub fn cargo_cmd(&self, crate_dir: impl AsRef<OsStr>, cmd: &str) -> Cmd<'_> {
        let MiriEnv { toolchain, cargo_extra_flags, .. } = self;
        let manifest_path = path!(self.miri_dir / crate_dir.as_ref() / "Cargo.toml");
        cmd!(
            self.sh,
            "cargo +{toolchain} {cmd} {cargo_extra_flags...} --manifest-path {manifest_path}"
        )
    }

    pub fn install_to_sysroot(
        &self,
        path: impl AsRef<OsStr>,
        args: impl IntoIterator<Item = impl AsRef<OsStr>>,
    ) -> Result<()> {
        let MiriEnv { sysroot, toolchain, cargo_extra_flags, .. } = self;
        let path = path!(self.miri_dir / path.as_ref());
        // Install binaries to the miri toolchain's `sysroot` so they do not interact with other toolchains.
        // (Not using `cargo_cmd` as `install` is special and doesn't use `--manifest-path`.)
        cmd!(self.sh, "cargo +{toolchain} install {cargo_extra_flags...} --path {path} --force --root {sysroot} {args...}").run()?;
        Ok(())
    }

    pub fn build(&self, crate_dir: impl AsRef<OsStr>, args: &[String], quiet: bool) -> Result<()> {
        let quiet_flag = if quiet { Some("--quiet") } else { None };
        // We build all targets, since building *just* the bin target doesnot include
        // `dev-dependencies` and that changes feature resolution. This also gets us more
        // parallelism in `./miri test` as we build Miri and its tests together.
        let mut cmd =
            self.cargo_cmd(crate_dir, "build").args(&["--all-targets"]).args(quiet_flag).args(args);
        cmd.set_quiet(quiet);
        cmd.run()?;
        Ok(())
    }

    /// Returns the path to the main crate binary. Assumes that `build` has been called before.
    pub fn build_get_binary(&self, crate_dir: impl AsRef<OsStr>) -> Result<PathBuf> {
        let cmd =
            self.cargo_cmd(crate_dir, "build").args(&["--all-targets", "--message-format=json"]);
        let output = cmd.output()?;
        let mut bin = None;
        for line in output.stdout.lines() {
            let line = line?;
            if line.starts_with("{") {
                let json: serde_json::Value = serde_json::from_str(&line)?;
                if json["reason"] == "compiler-artifact"
                    && !json["profile"]["test"].as_bool().unwrap()
                    && !json["executable"].is_null()
                {
                    if bin.is_some() {
                        bail!("found two binaries in cargo output");
                    }
                    bin = Some(PathBuf::from(json["executable"].as_str().unwrap()))
                }
            }
        }
        bin.ok_or_else(|| anyhow!("found no binary in cargo output"))
    }

    pub fn check(&self, crate_dir: impl AsRef<OsStr>, args: &[String]) -> Result<()> {
        self.cargo_cmd(crate_dir, "check").arg("--all-targets").args(args).run()?;
        Ok(())
    }

    pub fn doc(&self, crate_dir: impl AsRef<OsStr>, args: &[String]) -> Result<()> {
        self.cargo_cmd(crate_dir, "doc").args(args).run()?;
        Ok(())
    }

    pub fn clippy(&self, crate_dir: impl AsRef<OsStr>, args: &[String]) -> Result<()> {
        self.cargo_cmd(crate_dir, "clippy").arg("--all-targets").args(args).run()?;
        Ok(())
    }

    pub fn test(&self, crate_dir: impl AsRef<OsStr>, args: &[String]) -> Result<()> {
        self.cargo_cmd(crate_dir, "test").args(args).run()?;
        Ok(())
    }

    /// Receives an iterator of files.
    /// Will format each file with the miri rustfmt config.
    /// Does not recursively format modules.
    pub fn format_files(
        &self,
        files: impl Iterator<Item = Result<PathBuf, walkdir::Error>>,
        config_path: &Path,
        flags: &[String],
    ) -> anyhow::Result<()> {
        use itertools::Itertools;

        let mut first = true;

        // Format in batches as not all our files fit into Windows' command argument limit.
        for batch in &files.chunks(256) {
            // Build base command.
            let toolchain = &self.toolchain;
            let mut cmd = cmd!(
                self.sh,
                "rustfmt +{toolchain} --edition=2021 --config-path {config_path} --unstable-features --skip-children {flags...}"
            );
            if first {
                // Log an abbreviating command, and only once.
                eprintln!("$ {cmd} ...");
                first = false;
            }
            // Add files.
            for file in batch {
                // Make it a relative path so that on platforms with extremely tight argument
                // limits (like Windows), we become immune to someone cloning the repo
                // 50 directories deep.
                let file = file?;
                let file = file.strip_prefix(&self.miri_dir)?;
                cmd = cmd.arg(file);
            }

            // Run rustfmt.
            // We want our own error message, repeating the command is too much.
            cmd.quiet().run().map_err(|_| anyhow!("`rustfmt` failed"))?;
        }

        Ok(())
    }

    /// Run the given closure many times in parallel with access to the shell, once for each value in the `range`.
    pub fn run_many_times(
        &self,
        range: Range<u32>,
        run: impl Fn(&Self, u32) -> Result<()> + Sync,
    ) -> Result<()> {
        // `next` is atomic so threads can concurrently fetch their next value to run.
        let next = AtomicU32::new(range.start);
        let end = range.end; // exclusive!
        let failed = AtomicBool::new(false);
        thread::scope(|s| {
            let mut handles = Vec::new();
            // Spawn one worker per core.
            for _ in 0..thread::available_parallelism()?.get() {
                // Create a copy of the environment for this thread.
                let local_miri = self.clone();
                let handle = s.spawn(|| -> Result<()> {
                    let local_miri = local_miri; // move the copy into this thread.
                    // Each worker thread keeps asking for numbers until we're all done.
                    loop {
                        let cur = next.fetch_add(1, Ordering::Relaxed);
                        if cur >= end {
                            // We hit the upper limit and are done.
                            break;
                        }
                        // Run the command with this seed.
                        run(&local_miri, cur).inspect_err(|_| {
                            // If we failed, tell everyone about this.
                            failed.store(true, Ordering::Relaxed);
                        })?;
                        // Check if some other command failed (in which case we'll stop as well).
                        if failed.load(Ordering::Relaxed) {
                            return Ok(());
                        }
                    }
                    Ok(())
                });
                handles.push(handle);
            }
            // Wait for all workers to be done.
            for handle in handles {
                handle.join().unwrap()?;
            }
            // If all workers succeeded, we can't have failed.
            assert!(!failed.load(Ordering::Relaxed));
            Ok(())
        })
    }
}
