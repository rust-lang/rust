//! `./x.py clean`
//!
//! Responsible for cleaning out a build directory of all old and stale
//! artifacts to prepare for a fresh build. Currently doesn't remove the
//! `build/cache` directory (download cache) or the `build/$target/llvm`
//! directory unless the `--all` flag is present.

use std::fs;
use std::io::{self, ErrorKind};
use std::path::Path;

use crate::core::builder::{Builder, RunConfig, ShouldRun, Step, crate_description};
use crate::utils::build_stamp::BuildStamp;
use crate::utils::helpers::t;
use crate::{Build, Compiler, Kind, Mode, Subcommand};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CleanAll {}

impl Step for CleanAll {
    const DEFAULT: bool = true;
    type Output = ();

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(CleanAll {})
    }

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let Subcommand::Clean { all, stage } = builder.config.cmd else {
            unreachable!("wrong subcommand?")
        };

        if all && stage.is_some() {
            panic!("--all and --stage can't be used at the same time for `x clean`");
        }

        clean(builder.build, all, stage)
    }

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never() // handled by DEFAULT
    }
}

macro_rules! clean_crate_tree {
    ( $( $name:ident, $mode:path, $root_crate:literal);+ $(;)? ) => { $(
        #[derive(Debug, Clone, PartialEq, Eq, Hash)]
        pub struct $name {
            compiler: Compiler,
            crates: Vec<String>,
        }

        impl Step for $name {
            type Output = ();

            fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                let crates = run.builder.in_tree_crates($root_crate, None);
                run.crates(crates)
            }

            fn make_run(run: RunConfig<'_>) {
                let builder = run.builder;
                let compiler = builder.compiler(builder.top_stage, run.target);
                builder.ensure(Self { crates: run.cargo_crates_in_set(), compiler });
            }

            fn run(self, builder: &Builder<'_>) -> Self::Output {
                let compiler = self.compiler;
                let target = compiler.host;
                let mut cargo = builder.bare_cargo(compiler, $mode, target, Kind::Clean);

                // Since https://github.com/rust-lang/rust/pull/111076 enables
                // unstable cargo feature (`public-dependency`), we need to ensure
                // that unstable features are enabled before reading libstd Cargo.toml.
                cargo.env("RUSTC_BOOTSTRAP", "1");

                for krate in &*self.crates {
                    cargo.arg("-p");
                    cargo.arg(krate);
                }

                builder.info(&format!(
                    "Cleaning{} stage{} {} artifacts ({} -> {})",
                    crate_description(&self.crates), compiler.stage, stringify!($name).to_lowercase(), &compiler.host, target,
                ));

                // NOTE: doesn't use `run_cargo` because we don't want to save a stamp file,
                // and doesn't use `stream_cargo` to avoid passing `--message-format` which `clean` doesn't accept.
                cargo.run(builder);
            }
        }
    )+ }
}

clean_crate_tree! {
    Rustc, Mode::Rustc, "rustc-main";
    Std, Mode::Std, "sysroot";
}

fn clean(build: &Build, all: bool, stage: Option<u32>) {
    if build.config.dry_run() {
        return;
    }

    rm_rf("tmp".as_ref());

    // Clean the entire build directory
    if all {
        rm_rf(&build.out);
        return;
    }

    // Clean the target stage artifacts
    if let Some(stage) = stage {
        clean_specific_stage(build, stage);
        return;
    }

    // Follow the default behaviour
    clean_default(build);
}

fn clean_specific_stage(build: &Build, stage: u32) {
    for host in &build.hosts {
        let entries = match build.out.join(host).read_dir() {
            Ok(iter) => iter,
            Err(_) => continue,
        };

        for entry in entries {
            let entry = t!(entry);
            let stage_prefix = format!("stage{stage}");

            // if current entry is not related with the target stage, continue
            if !entry.file_name().to_str().unwrap_or("").contains(&stage_prefix) {
                continue;
            }

            let path = t!(entry.path().canonicalize());
            rm_rf(&path);
        }
    }
}

fn clean_default(build: &Build) {
    rm_rf(&build.out.join("tmp"));
    rm_rf(&build.out.join("dist"));
    rm_rf(&build.out.join("bootstrap").join(".last-warned-change-id"));
    rm_rf(&build.out.join("bootstrap-shims-dump"));
    rm_rf(BuildStamp::new(&build.out).with_prefix("rustfmt").path());

    let mut hosts: Vec<_> = build.hosts.iter().map(|t| build.out.join(t)).collect();
    // After cross-compilation, artifacts of the host architecture (which may differ from build.host)
    // might not get removed.
    // Adding its path (linked one for easier accessibility) will solve this problem.
    hosts.push(build.out.join("host"));

    for host in hosts {
        let entries = match host.read_dir() {
            Ok(iter) => iter,
            Err(_) => continue,
        };

        for entry in entries {
            let entry = t!(entry);
            if entry.file_name().to_str() == Some("llvm") {
                continue;
            }
            let path = t!(entry.path().canonicalize());
            rm_rf(&path);
        }
    }
}

fn rm_rf(path: &Path) {
    match path.symlink_metadata() {
        Err(e) => {
            if e.kind() == ErrorKind::NotFound {
                return;
            }
            panic!("failed to get metadata for file {}: {}", path.display(), e);
        }
        Ok(metadata) => {
            if metadata.file_type().is_file() || metadata.file_type().is_symlink() {
                do_op(path, "remove file", |p| match fs::remove_file(p) {
                    #[cfg(windows)]
                    Err(e)
                        if e.kind() == std::io::ErrorKind::PermissionDenied
                            && p.file_name().and_then(std::ffi::OsStr::to_str)
                                == Some("bootstrap.exe") =>
                    {
                        eprintln!("WARNING: failed to delete '{}'.", p.display());
                        Ok(())
                    }
                    r => r,
                });

                return;
            }

            for file in t!(fs::read_dir(path)) {
                rm_rf(&t!(file).path());
            }

            do_op(path, "remove dir", |p| match fs::remove_dir(p) {
                // Check for dir not empty on Windows
                #[cfg(windows)]
                Err(e) if e.kind() == ErrorKind::DirectoryNotEmpty => Ok(()),
                r => r,
            });
        }
    };
}

fn do_op<F>(path: &Path, desc: &str, mut f: F)
where
    F: FnMut(&Path) -> io::Result<()>,
{
    match f(path) {
        Ok(()) => {}
        // On windows we can't remove a readonly file, and git will often clone files as readonly.
        // As a result, we have some special logic to remove readonly files on windows.
        // This is also the reason that we can't use things like fs::remove_dir_all().
        #[cfg(windows)]
        Err(ref e) if e.kind() == ErrorKind::PermissionDenied => {
            let m = t!(path.symlink_metadata());
            let mut p = m.permissions();
            // this os not unix, so clippy gives FP
            #[expect(clippy::permissions_set_readonly_false)]
            p.set_readonly(false);
            t!(fs::set_permissions(path, p));
            f(path).unwrap_or_else(|e| {
                // Delete symlinked directories on Windows
                if m.file_type().is_symlink() && path.is_dir() && fs::remove_dir(path).is_ok() {
                    return;
                }
                panic!("failed to {} {}: {}", desc, path.display(), e);
            });
        }
        Err(e) => {
            panic!("failed to {} {}: {}", desc, path.display(), e);
        }
    }
}
