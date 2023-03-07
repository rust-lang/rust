//! Implementation of `make clean` in rustbuild.
//!
//! Responsible for cleaning out a build directory of all old and stale
//! artifacts to prepare for a fresh build. Currently doesn't remove the
//! `build/cache` directory (download cache) or the `build/$target/llvm`
//! directory unless the `--all` flag is present.

use std::fs;
use std::io::{self, ErrorKind};
use std::path::Path;

use crate::builder::{crate_description, Builder, RunConfig, ShouldRun, Step};
use crate::cache::Interned;
use crate::util::t;
use crate::{Build, Compiler, Mode, Subcommand};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CleanAll {}

impl Step for CleanAll {
    const DEFAULT: bool = true;
    type Output = ();

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(CleanAll {})
    }

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let Subcommand::Clean { all, .. } = builder.config.cmd else { unreachable!("wrong subcommand?") };
        clean_default(builder.build, all)
    }

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never() // handled by DEFAULT
    }
}

macro_rules! clean_crate_tree {
    ( $( $name:ident, $mode:path, $root_crate:literal);+ $(;)? ) => { $(
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name {
            compiler: Compiler,
            crates: Interned<Vec<String>>,
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
                let mut cargo = builder.bare_cargo(compiler, $mode, target, "clean");
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
                builder.run(&mut cargo);
            }
        }
    )+ }
}

clean_crate_tree! {
    Rustc, Mode::Rustc, "rustc-main";
    Std, Mode::Std, "test";
}

fn clean_default(build: &Build, all: bool) {
    rm_rf("tmp".as_ref());

    if all {
        rm_rf(&build.out);
    } else {
        rm_rf(&build.out.join("tmp"));
        rm_rf(&build.out.join("dist"));
        rm_rf(&build.out.join("bootstrap"));
        rm_rf(&build.out.join("rustfmt.stamp"));

        for host in &build.hosts {
            let entries = match build.out.join(host.triple).read_dir() {
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
                do_op(path, "remove file", |p| {
                    fs::remove_file(p).or_else(|e| {
                        // Work around the fact that we cannot
                        // delete an executable while it runs on Windows.
                        #[cfg(windows)]
                        if e.kind() == std::io::ErrorKind::PermissionDenied
                            && p.file_name().and_then(std::ffi::OsStr::to_str)
                                == Some("bootstrap.exe")
                        {
                            eprintln!("warning: failed to delete '{}'.", p.display());
                            return Ok(());
                        }
                        Err(e)
                    })
                });
                return;
            }

            for file in t!(fs::read_dir(path)) {
                rm_rf(&t!(file).path());
            }
            do_op(path, "remove dir", |p| {
                fs::remove_dir(p).or_else(|e| {
                    // Check for dir not empty on Windows
                    // FIXME: Once `ErrorKind::DirectoryNotEmpty` is stabilized,
                    // match on `e.kind()` instead.
                    #[cfg(windows)]
                    if e.raw_os_error() == Some(145) {
                        return Ok(());
                    }

                    Err(e)
                })
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
        Err(ref e) if cfg!(windows) && e.kind() == ErrorKind::PermissionDenied => {
            let m = t!(path.symlink_metadata());
            let mut p = m.permissions();
            p.set_readonly(false);
            t!(fs::set_permissions(path, p));
            f(path).unwrap_or_else(|e| {
                // Delete symlinked directories on Windows
                #[cfg(windows)]
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
