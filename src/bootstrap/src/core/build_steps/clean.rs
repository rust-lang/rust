//! `./x.py clean`
//!
//! Responsible for cleaning out a build directory of all old and stale
//! artifacts to prepare for a fresh build. Currently doesn't remove the
//! `build/cache` directory (download cache) or the `build/$target/llvm`
//! directory unless the `--all` flag is present.

use std::fs;
use std::path::Path;

use crate::core::builder::{crate_description, Builder, RunConfig, ShouldRun, Step};
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

    remove_dir_recursive("tmp");

    // Clean the entire build directory
    if all {
        remove_dir_recursive(&build.out);
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
            let stage_prefix = format!("stage{}", stage);

            // if current entry is not related with the target stage, continue
            if !entry.file_name().to_str().unwrap_or("").contains(&stage_prefix) {
                continue;
            }

            let path = t!(entry.path().canonicalize());
            remove_dir_recursive(&path);
        }
    }
}

fn clean_default(build: &Build) {
    remove_dir_recursive(build.out.join("tmp"));
    remove_dir_recursive(build.out.join("dist"));
    remove_dir_recursive(build.out.join("bootstrap").join(".last-warned-change-id"));
    remove_dir_recursive(build.out.join("bootstrap-shims-dump"));
    remove_dir_recursive(build.out.join("rustfmt.stamp"));

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
            remove_dir_recursive(&path);
        }
    }
}

/// Wrapper for [`std::fs::remove_dir_all`] that panics on failure and prints the `path` we failed
/// on.
fn remove_dir_recursive<P: AsRef<Path>>(path: P) {
    let path = path.as_ref();
    if let Err(e) = fs::remove_dir_all(path) {
        panic!("failed to `remove_dir_all` at `{}`: {e}", path.display());
    }
}
