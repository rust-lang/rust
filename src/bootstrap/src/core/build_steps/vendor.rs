use std::path::PathBuf;

use crate::core::build_steps::tool::SUBMODULES_FOR_RUSTBOOK;
use crate::core::builder::{Builder, RunConfig, ShouldRun, Step};
use crate::utils::exec::command;

/// List of default paths used for vendoring for `x vendor` and dist tarballs.
pub fn default_paths_to_vendor(builder: &Builder<'_>) -> Vec<PathBuf> {
    let mut paths = vec![];
    for p in [
        "src/tools/cargo/Cargo.toml",
        "src/tools/rust-analyzer/Cargo.toml",
        "compiler/rustc_codegen_cranelift/Cargo.toml",
        "compiler/rustc_codegen_gcc/Cargo.toml",
        "library/Cargo.toml",
        "src/bootstrap/Cargo.toml",
        "src/tools/rustbook/Cargo.toml",
        "src/tools/rustc-perf/Cargo.toml",
        "src/tools/opt-dist/Cargo.toml",
        "src/doc/book/packages/trpl/Cargo.toml",
    ] {
        paths.push(builder.src.join(p));
    }

    paths
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub(crate) struct Vendor {
    sync_args: Vec<PathBuf>,
    versioned_dirs: bool,
    root_dir: PathBuf,
}

impl Step for Vendor {
    type Output = ();
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.alias("placeholder").default_condition(true)
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Vendor {
            sync_args: run.builder.config.cmd.vendor_sync_args(),
            versioned_dirs: run.builder.config.cmd.vendor_versioned_dirs(),
            root_dir: run.builder.src.clone(),
        });
    }

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let mut cmd = command(&builder.initial_cargo);
        cmd.arg("vendor");

        if self.versioned_dirs {
            cmd.arg("--versioned-dirs");
        }

        // These submodules must be present for `x vendor` to work.
        for submodule in SUBMODULES_FOR_RUSTBOOK.iter().chain(["src/tools/cargo"].iter()) {
            builder.build.require_submodule(submodule, None);
        }

        // Sync these paths by default.
        for p in default_paths_to_vendor(builder) {
            cmd.arg("--sync").arg(p);
        }

        // Also sync explicitly requested paths.
        for sync_arg in self.sync_args {
            cmd.arg("--sync").arg(sync_arg);
        }

        // Will read the libstd Cargo.toml
        // which uses the unstable `public-dependency` feature.
        cmd.env("RUSTC_BOOTSTRAP", "1");

        cmd.current_dir(self.root_dir);

        cmd.run(builder);
    }
}
