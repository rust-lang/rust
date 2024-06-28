use crate::core::builder::{Builder, RunConfig, ShouldRun, Step};
use std::path::{Path, PathBuf};
use std::process::Command;

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
        let mut cmd = Command::new(&builder.initial_cargo);
        cmd.arg("vendor");

        if self.versioned_dirs {
            cmd.arg("--versioned-dirs");
        }

        // cargo submodule must be present for `x vendor` to work.
        builder.build.update_submodule(Path::new("src/tools/cargo"));

        // Sync these paths by default.
        for p in [
            "src/tools/cargo/Cargo.toml",
            "src/tools/rust-analyzer/Cargo.toml",
            "compiler/rustc_codegen_cranelift/Cargo.toml",
            "compiler/rustc_codegen_gcc/Cargo.toml",
            "src/bootstrap/Cargo.toml",
        ] {
            cmd.arg("--sync").arg(builder.src.join(p));
        }

        // Also sync explicitly requested paths.
        for sync_arg in self.sync_args {
            cmd.arg("--sync").arg(sync_arg);
        }

        // Will read the libstd Cargo.toml
        // which uses the unstable `public-dependency` feature.
        cmd.env("RUSTC_BOOTSTRAP", "1");

        cmd.current_dir(self.root_dir);

        builder.run(&mut cmd);
    }
}
