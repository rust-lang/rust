//! In some cases, parts of bootstrap need to change part of a target spec just for one or a few
//! steps. Adding these targets to rustc proper would "leak" this implementation detail of
//! bootstrap, and would make it more complex to apply additional changes if the need arises.
//!
//! To address that problem, this module implements support for "synthetic targets". Synthetic
//! targets are custom target specs generated using builtin target specs as their base. You can use
//! one of the target specs already defined in this module, or create new ones by adding a new step
//! that calls create_synthetic_target.

use crate::Compiler;
use crate::core::builder::{Builder, ShouldRun, Step};
use crate::core::config::TargetSelection;
use crate::utils::exec::command;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct MirOptPanicAbortSyntheticTarget {
    pub(crate) compiler: Compiler,
    pub(crate) base: TargetSelection,
}

impl Step for MirOptPanicAbortSyntheticTarget {
    type Output = TargetSelection;
    const DEFAULT: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    fn run(self, builder: &Builder<'_>) -> Self::Output {
        create_synthetic_target(builder, self.compiler, "miropt-abort", self.base, |spec| {
            spec.insert("panic-strategy".into(), "abort".into());
        })
    }
}

fn create_synthetic_target(
    builder: &Builder<'_>,
    compiler: Compiler,
    suffix: &str,
    base: TargetSelection,
    customize: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>),
) -> TargetSelection {
    if base.contains("synthetic") {
        // This check is not strictly needed, but nothing currently needs recursive synthetic
        // targets. If the need arises, removing this in the future *SHOULD* be safe.
        panic!("cannot create synthetic targets with other synthetic targets as their base");
    }

    let name = format!("{base}-synthetic-{suffix}");
    let path = builder.out.join("synthetic-target-specs").join(format!("{name}.json"));
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();

    if builder.config.dry_run() {
        std::fs::write(&path, b"dry run\n").unwrap();
        return TargetSelection::create_synthetic(&name, path.to_str().unwrap());
    }

    let mut cmd = command(builder.rustc(compiler));
    cmd.arg("--target").arg(base.rustc_target_arg());
    cmd.args(["-Zunstable-options", "--print", "target-spec-json"]);

    // If `rust.channel` is set to either beta or stable, rustc will complain that
    // we cannot use nightly features. So `RUSTC_BOOTSTRAP` is needed here.
    cmd.env("RUSTC_BOOTSTRAP", "1");

    let output = cmd.run_capture(builder).stdout();
    let mut spec: serde_json::Value = serde_json::from_slice(output.as_bytes()).unwrap();
    let spec_map = spec.as_object_mut().unwrap();

    // The `is-builtin` attribute of a spec needs to be removed, otherwise rustc will complain.
    spec_map.remove("is-builtin");

    customize(spec_map);

    std::fs::write(&path, serde_json::to_vec_pretty(&spec).unwrap()).unwrap();
    TargetSelection::create_synthetic(&name, path.to_str().unwrap())
}
