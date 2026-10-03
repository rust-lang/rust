//! Compilation of TPDE shared library, tpde-plugin.so, for use as an alternative
//! LLVM backend in rustc_codegen_llvm.
//!
//! Native projects like TPDE unfortunately aren't suited just yet for
//! compilation in build scripts that Cargo has. This is because the
//! compilation takes a long time but also because we don't want to
//! compile TPDE 3 times as part of a normal bootstrap (we want it cached).
//!
//! TPDE is always built from source, as it's small enough that downloading it from CI
//! is more trouble than it's worth.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::core::build_steps::compile::strip_debug;
use crate::core::build_steps::llvm::{
    CcFlags, LdFlags, Llvm, configure_cmake, try_link_with_in_tree_lld,
};
use crate::core::builder::{Builder, CommandLineStep, Kind, RunConfig, ShouldRun, StepMetadata};
use crate::core::config::{Config, TargetSelection};
use crate::utils::build_stamp::{BuildStamp, generate_smart_stamp_hash};
use crate::utils::helpers::{self, t};

/// Result of building TPDE artifacts.
///
/// Currently only tpde-plugin.so is consumed elsewhere.
#[derive(Clone)]
pub struct TpdeOutput {
    tpde_root_dir: PathBuf,
    tpde_plugin_path: PathBuf,
}

impl TpdeOutput {
    /// Path to tpde-plugin.so
    pub fn plugin_path(&self) -> &Path {
        &self.tpde_plugin_path
    }
}

pub struct TpdeBuildInfo {
    stamp: BuildStamp,
    output: TpdeOutput,
}

pub enum TpdeBuildStatus {
    AlreadyBuilt(TpdeOutput),
    ShouldBuild(TpdeBuildInfo),
}

/// Return build status of TPDE, considering only the locally built TPDE.
///
/// Calling this function should never attempt to checkout the TPDE submodule.
fn get_locally_built_tpde_build_status(
    builder: &Builder<'_>,
    target: TargetSelection,
) -> TpdeBuildStatus {
    let out_dir = tpde_output_dir(builder, target);

    let res = TpdeOutput {
        tpde_plugin_path: out_dir.join("build").join("tpde-llvm").join("tpde-plugin.so"),
        tpde_root_dir: out_dir,
    };

    static STAMP_HASH_MEMO: OnceLock<String> = OnceLock::new();
    let smart_stamp_hash = STAMP_HASH_MEMO.get_or_init(|| {
        generate_smart_stamp_hash(
            builder,
            &builder.config.src.join("src/tpde"),
            builder.in_tree_tpde_info.sha().unwrap_or_default(),
        )
    });

    // Rebuild if TPDE options change
    let stamp = BuildStamp::new(&res.tpde_root_dir)
        .with_prefix("tpde")
        .add_stamp(smart_stamp_hash)
        .add_stamp(builder.config.tpde_assertions)
        .add_stamp(builder.config.tpde_optimize)
        .add_stamp(builder.config.tpde_release_debuginfo);

    if stamp.is_up_to_date() {
        if smart_stamp_hash.is_empty() {
            builder.info(
                "Could not determine the TPDE submodule commit hash. \
                     Assuming that a TPDE rebuild is not necessary.",
            );
            builder.info(&format!(
                "To force TPDE to rebuild, remove the file `{}`",
                stamp.path().display()
            ));
        }
        return TpdeBuildStatus::AlreadyBuilt(res);
    }

    TpdeBuildStatus::ShouldBuild(TpdeBuildInfo { stamp, output: res })
}

/// Output directory of *locally built* TPDE for the given `target`.
/// Should only be used within this module, when building TPDE.
/// Otherwise, you should ensure the `Tpde` step and read its root directory.
fn tpde_output_dir(builder: &Builder<'_>, target: TargetSelection) -> PathBuf {
    builder.config.out.join(target).join("tpde")
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Tpde {
    pub target: TargetSelection,
}

// Most of this logic is copied and simplified from llvm::Llvm.
impl CommandLineStep for Tpde {
    type Output = TpdeOutput;

    const IS_HOST: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tpde").alias("tpde")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Tpde { target: run.target });
    }

    /// Compile TPDE for `target`.
    fn run(self, builder: &Builder<'_>) -> TpdeOutput {
        let target = self.target;
        let llvm = builder.ensure(Llvm { target });
        if fs::read_dir(llvm.cmake_dir()).into_iter().flatten().flat_map(Result::ok).count() == 0 {
            // If it can't find LLVMConfig.cmake, TPDE falls back to any other LLVM in the system path
            builder.info(&format!(
                "WARNING: {:?} is empty, which will cause TPDE to be built with the system LLVM. \
                This may be because you are using CI LLVM, which does not distribute the required CMake files.",
                llvm.cmake_dir()
            ));
        }
        builder.config.update_submodule("src/tpde");
        // If TPDE has already been built, we avoid building it again.
        let TpdeBuildInfo { stamp, output } =
            match get_locally_built_tpde_build_status(builder, target) {
                TpdeBuildStatus::AlreadyBuilt(p) => return p,
                TpdeBuildStatus::ShouldBuild(m) => m,
            };

        let _guard = builder.msg_unstaged(Kind::Build, "TPDE", target);
        t!(stamp.remove());
        let _time = helpers::timeit(builder);
        t!(fs::create_dir_all(&output.tpde_root_dir));

        let mut cfg = cmake::Config::new(builder.src.join("src/tpde"));

        let profile = get_tpde_profile(&builder.config);
        let assertions = if builder.config.tpde_assertions { "ON" } else { "OFF" };

        cfg.out_dir(&output.tpde_root_dir)
            .profile(profile)
            .define("TPDE_ENABLE_ASSERTIONS", assertions)
            .define("TPDE_ENABLE_LLVM", "ON")
            .define("TPDE_ENABLE_ENCODEGEN", "ON")
            .define("TPDE_INCLUDE_TESTS", "OFF")
            .define("LLVM_DIR", llvm.cmake_dir());

        let mut ldflags = LdFlags::default();
        try_link_with_in_tree_lld(builder, target, &llvm, &mut cfg, &mut ldflags);
        configure_cmake(builder, target, &mut cfg, true, ldflags, CcFlags::default(), &[]);

        if !builder.config.is_host_target(target) {
            panic!("bootstrap does not support cross-compiling TPDE yet");
        }

        if builder.config.dry_run() {
            return output;
        }

        cfg.build();

        // When building TPDE as a shared library on linux, it can contain unexpected debuginfo:
        // some can come from the C++ standard library. Unless we're explicitly requesting TPDE to
        // be built with debuginfo, strip it away after the fact, to make dist artifacts smaller.
        if builder.config.tpde_optimize && !builder.config.tpde_release_debuginfo {
            // If the shared library exists in TPDE's `/build/tpde-llvm` folder, strip its
            // debuginfo.
            strip_debug(builder, target, output.plugin_path());
        }

        t!(stamp.write());

        output
    }

    fn metadata(&self) -> Option<StepMetadata> {
        Some(StepMetadata::build("tpde", self.target))
    }
}

fn get_tpde_profile(config: &Config) -> &'static str {
    match (config.tpde_optimize, config.tpde_release_debuginfo) {
        (false, _) => "Debug",
        (true, false) => "Release",
        (true, true) => "RelWithDebInfo",
    }
}
