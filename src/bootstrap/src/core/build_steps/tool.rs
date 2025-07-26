//! This module handles building and managing various tools in bootstrap
//! build system.
//!
//! **What It Does**
//! - Defines how tools are built, configured and installed.
//! - Manages tool dependencies and build steps.
//! - Copies built tool binaries to the correct locations.
//!
//! Each Rust tool **MUST** utilize `ToolBuild` inside their `Step` logic,
//! return `ToolBuildResult` and should never prepare `cargo` invocations manually.

use std::ffi::OsStr;
use std::path::PathBuf;
use std::{env, fs};

#[cfg(feature = "tracing")]
use tracing::instrument;

use crate::core::build_steps::compile::is_lto_stage;
use crate::core::build_steps::toolstate::ToolState;
use crate::core::build_steps::{compile, llvm};
use crate::core::builder;
use crate::core::builder::{
    Builder, Cargo as CargoCommand, RunConfig, ShouldRun, Step, StepMetadata, cargo_profile_var,
};
use crate::core::config::{DebuginfoLevel, RustcLto, TargetSelection};
use crate::utils::exec::{BootstrapCommand, command};
use crate::utils::helpers::{add_dylib_path, exe, t};
use crate::{Compiler, FileType, Kind, Mode, gha};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum SourceType {
    InTree,
    Submodule,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum ToolArtifactKind {
    Binary,
    Library,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct ToolBuild {
    /// Compiler that will build this tool.
    build_compiler: Compiler,
    target: TargetSelection,
    tool: &'static str,
    path: &'static str,
    mode: Mode,
    source_type: SourceType,
    extra_features: Vec<String>,
    /// Nightly-only features that are allowed (comma-separated list).
    allow_features: &'static str,
    /// Additional arguments to pass to the `cargo` invocation.
    cargo_args: Vec<String>,
    /// Whether the tool builds a binary or a library.
    artifact_kind: ToolArtifactKind,
}

impl Builder<'_> {
    #[track_caller]
    pub(crate) fn msg_tool(
        &self,
        kind: Kind,
        mode: Mode,
        tool: &str,
        build_stage: u32,
        host: &TargetSelection,
        target: &TargetSelection,
    ) -> Option<gha::Group> {
        match mode {
            // depends on compiler stage, different to host compiler
            Mode::ToolRustc => self.msg_sysroot_tool(
                kind,
                build_stage,
                format_args!("tool {tool}"),
                *host,
                *target,
            ),
            // doesn't depend on compiler, same as host compiler
            _ => self.msg(kind, build_stage, format_args!("tool {tool}"), *host, *target),
        }
    }
}

/// Result of the tool build process. Each `Step` in this module is responsible
/// for using this type as `type Output = ToolBuildResult;`
#[derive(Clone)]
pub struct ToolBuildResult {
    /// Artifact path of the corresponding tool that was built.
    pub tool_path: PathBuf,
    /// Compiler used to build the tool. For non-`ToolRustc` tools this is equal to `target_compiler`.
    /// For `ToolRustc` this is one stage before of the `target_compiler`.
    pub build_compiler: Compiler,
    /// Target compiler passed to `Step`.
    pub target_compiler: Compiler,
}

impl Step for ToolBuild {
    type Output = ToolBuildResult;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    /// Builds a tool in `src/tools`
    ///
    /// This will build the specified tool with the specified `host` compiler in
    /// `stage` into the normal cargo output directory.
    fn run(mut self, builder: &Builder<'_>) -> ToolBuildResult {
        let target = self.target;
        let mut tool = self.tool;
        let path = self.path;

        let target_compiler = self.build_compiler;
        self.build_compiler = if self.mode == Mode::ToolRustc {
            get_tool_rustc_compiler(builder, self.build_compiler)
        } else {
            self.build_compiler
        };

        match self.mode {
            Mode::ToolRustc => {
                // If compiler was forced, its artifacts should have been prepared earlier.
                if !self.build_compiler.is_forced_compiler() {
                    builder.std(self.build_compiler, self.build_compiler.host);
                    builder.ensure(compile::Rustc::new(self.build_compiler, target));
                }
            }
            Mode::ToolStd => {
                // If compiler was forced, its artifacts should have been prepared earlier.
                if !self.build_compiler.is_forced_compiler() {
                    builder.std(self.build_compiler, target);
                }
            }
            Mode::ToolBootstrap | Mode::ToolTarget => {} // uses downloaded stage0 compiler libs
            _ => panic!("unexpected Mode for tool build"),
        }

        let mut cargo = prepare_tool_cargo(
            builder,
            self.build_compiler,
            self.mode,
            target,
            Kind::Build,
            path,
            self.source_type,
            &self.extra_features,
        );

        // The stage0 compiler changes infrequently and does not directly depend on code
        // in the current working directory. Therefore, caching it with sccache should be
        // useful.
        // This is only performed for non-incremental builds, as ccache cannot deal with these.
        if let Some(ref ccache) = builder.config.ccache
            && matches!(self.mode, Mode::ToolBootstrap)
            && !builder.config.incremental
        {
            cargo.env("RUSTC_WRAPPER", ccache);
        }

        // Rustc tools (miri, clippy, cargo, rustfmt, rust-analyzer)
        // could use the additional optimizations.
        if self.mode == Mode::ToolRustc && is_lto_stage(&self.build_compiler) {
            let lto = match builder.config.rust_lto {
                RustcLto::Off => Some("off"),
                RustcLto::Thin => Some("thin"),
                RustcLto::Fat => Some("fat"),
                RustcLto::ThinLocal => None,
            };
            if let Some(lto) = lto {
                cargo.env(cargo_profile_var("LTO", &builder.config), lto);
            }
        }

        if !self.allow_features.is_empty() {
            cargo.allow_features(self.allow_features);
        }

        cargo.args(self.cargo_args);

        let _guard = builder.msg_tool(
            Kind::Build,
            self.mode,
            self.tool,
            // A stage N tool is built with the stage N-1 compiler.
            self.build_compiler.stage + 1,
            &self.build_compiler.host,
            &self.target,
        );

        // we check this below
        let build_success = compile::stream_cargo(builder, cargo, vec![], &mut |_| {});

        builder.save_toolstate(
            tool,
            if build_success { ToolState::TestFail } else { ToolState::BuildFail },
        );

        if !build_success {
            crate::exit!(1);
        } else {
            // HACK(#82501): on Windows, the tools directory gets added to PATH when running tests, and
            // compiletest confuses HTML tidy with the in-tree tidy. Name the in-tree tidy something
            // different so the problem doesn't come up.
            if tool == "tidy" {
                tool = "rust-tidy";
            }
            let tool_path = match self.artifact_kind {
                ToolArtifactKind::Binary => {
                    copy_link_tool_bin(builder, self.build_compiler, self.target, self.mode, tool)
                }
                ToolArtifactKind::Library => builder
                    .cargo_out(self.build_compiler, self.mode, self.target)
                    .join(format!("lib{tool}.rlib")),
            };

            ToolBuildResult { tool_path, build_compiler: self.build_compiler, target_compiler }
        }
    }
}

#[expect(clippy::too_many_arguments)] // FIXME: reduce the number of args and remove this.
pub fn prepare_tool_cargo(
    builder: &Builder<'_>,
    compiler: Compiler,
    mode: Mode,
    target: TargetSelection,
    cmd_kind: Kind,
    path: &str,
    source_type: SourceType,
    extra_features: &[String],
) -> CargoCommand {
    let mut cargo = builder::Cargo::new(builder, compiler, mode, source_type, target, cmd_kind);

    let path = PathBuf::from(path);
    let dir = builder.src.join(&path);
    cargo.arg("--manifest-path").arg(dir.join("Cargo.toml"));

    let mut features = extra_features.to_vec();
    if builder.build.config.cargo_native_static {
        if path.ends_with("cargo")
            || path.ends_with("clippy")
            || path.ends_with("miri")
            || path.ends_with("rustfmt")
        {
            cargo.env("LIBZ_SYS_STATIC", "1");
        }
        if path.ends_with("cargo") {
            features.push("all-static".to_string());
        }
    }

    // build.tool.TOOL_NAME.features in bootstrap.toml allows specifying which features to enable
    // for a specific tool. `extra_features` instead is not controlled by the toml and provides
    // features that are always enabled for a specific tool (e.g. "in-rust-tree" for rust-analyzer).
    // Finally, `prepare_tool_cargo` above here might add more features to adapt the build
    // to the chosen flags (e.g. "all-static" for cargo if `cargo_native_static` is true).
    builder
        .config
        .tool
        .iter()
        .filter(|(tool_name, _)| path.file_name().and_then(OsStr::to_str) == Some(tool_name))
        .for_each(|(_, tool)| features.extend(tool.features.clone().unwrap_or_default()));

    // clippy tests need to know about the stage sysroot. Set them consistently while building to
    // avoid rebuilding when running tests.
    cargo.env("SYSROOT", builder.sysroot(compiler));

    // if tools are using lzma we want to force the build script to build its
    // own copy
    cargo.env("LZMA_API_STATIC", "1");

    // CFG_RELEASE is needed by rustfmt (and possibly other tools) which
    // import rustc-ap-rustc_attr which requires this to be set for the
    // `#[cfg(version(...))]` attribute.
    cargo.env("CFG_RELEASE", builder.rust_release());
    cargo.env("CFG_RELEASE_CHANNEL", &builder.config.channel);
    cargo.env("CFG_VERSION", builder.rust_version());
    cargo.env("CFG_RELEASE_NUM", &builder.version);
    cargo.env("DOC_RUST_LANG_ORG_CHANNEL", builder.doc_rust_lang_org_channel());

    if let Some(ref ver_date) = builder.rust_info().commit_date() {
        cargo.env("CFG_VER_DATE", ver_date);
    }

    if let Some(ref ver_hash) = builder.rust_info().sha() {
        cargo.env("CFG_VER_HASH", ver_hash);
    }

    if let Some(description) = &builder.config.description {
        cargo.env("CFG_VER_DESCRIPTION", description);
    }

    let info = builder.config.git_info(builder.config.omit_git_hash, &dir);
    if let Some(sha) = info.sha() {
        cargo.env("CFG_COMMIT_HASH", sha);
    }

    if let Some(sha_short) = info.sha_short() {
        cargo.env("CFG_SHORT_COMMIT_HASH", sha_short);
    }

    if let Some(date) = info.commit_date() {
        cargo.env("CFG_COMMIT_DATE", date);
    }

    if !features.is_empty() {
        cargo.arg("--features").arg(features.join(", "));
    }

    // Enable internal lints for clippy and rustdoc
    // NOTE: this doesn't enable lints for any other tools unless they explicitly add `#![warn(rustc::internal)]`
    // See https://github.com/rust-lang/rust/pull/80573#issuecomment-754010776
    //
    // NOTE: We unconditionally set this here to avoid recompiling tools between `x check $tool`
    // and `x test $tool` executions.
    // See https://github.com/rust-lang/rust/issues/116538
    cargo.rustflag("-Zunstable-options");

    // NOTE: The root cause of needing `-Zon-broken-pipe=kill` in the first place is because `rustc`
    // and `rustdoc` doesn't gracefully handle I/O errors due to usages of raw std `println!` macros
    // which panics upon encountering broken pipes. `-Zon-broken-pipe=kill` just papers over that
    // and stops rustc/rustdoc ICEing on e.g. `rustc --print=sysroot | false`.
    //
    // cargo explicitly does not want the `-Zon-broken-pipe=kill` paper because it does actually use
    // variants of `println!` that handles I/O errors gracefully. It's also a breaking change for a
    // spawn process not written in Rust, especially if the language default handler is not
    // `SIG_IGN`. Thankfully cargo tests will break if we do set the flag.
    //
    // For the cargo discussion, see
    // <https://rust-lang.zulipchat.com/#narrow/stream/246057-t-cargo/topic/Applying.20.60-Zon-broken-pipe.3Dkill.60.20flags.20in.20bootstrap.3F>.
    //
    // For the rustc discussion, see
    // <https://rust-lang.zulipchat.com/#narrow/stream/131828-t-compiler/topic/Internal.20lint.20for.20raw.20.60print!.60.20and.20.60println!.60.3F>
    // for proper solutions.
    if !path.ends_with("cargo") {
        // Use an untracked env var `FORCE_ON_BROKEN_PIPE_KILL` here instead of `RUSTFLAGS`.
        // `RUSTFLAGS` is tracked by cargo. Conditionally omitting `-Zon-broken-pipe=kill` from
        // `RUSTFLAGS` causes unnecessary tool rebuilds due to cache invalidation from building e.g.
        // cargo *without* `-Zon-broken-pipe=kill` but then rustdoc *with* `-Zon-broken-pipe=kill`.
        cargo.env("FORCE_ON_BROKEN_PIPE_KILL", "-Zon-broken-pipe=kill");
    }

    cargo
}

/// Handle stage-off logic for `ToolRustc` tools when necessary.
pub(crate) fn get_tool_rustc_compiler(
    builder: &Builder<'_>,
    target_compiler: Compiler,
) -> Compiler {
    if target_compiler.is_forced_compiler() {
        return target_compiler;
    }

    if builder.download_rustc() && target_compiler.stage == 1 {
        // We shouldn't drop to stage0 compiler when using CI rustc.
        return builder.compiler(1, builder.config.host_target);
    }

    // Similar to `compile::Assemble`, build with the previous stage's compiler. Otherwise
    // we'd have stageN/bin/rustc and stageN/bin/$rustc_tool be effectively different stage
    // compilers, which isn't what we want. Rustc tools should be linked in the same way as the
    // compiler it's paired with, so it must be built with the previous stage compiler.
    builder.compiler(target_compiler.stage.saturating_sub(1), builder.config.host_target)
}

/// Determines how to build a `ToolTarget`, i.e. which compiler should be used to compile it.
/// The compiler stage is automatically bumped if we need to cross-compile a stage 1 tool.
pub enum ToolTargetBuildMode {
    /// Build the tool using rustc that corresponds to the selected CLI stage.
    Build(TargetSelection),
    /// Build the tool so that it can be attached to the sysroot of the passed compiler.
    /// Since we always dist stage 2+, the compiler that builds the tool in this case has to be
    /// stage 1+.
    Dist(Compiler),
}

/// Returns compiler that is able to compile a `ToolTarget` tool with the given `mode`.
pub(crate) fn get_tool_target_compiler(
    builder: &Builder<'_>,
    mode: ToolTargetBuildMode,
) -> Compiler {
    let (target, build_compiler_stage) = match mode {
        ToolTargetBuildMode::Build(target) => {
            assert!(builder.top_stage > 0);
            // If we want to build a stage N tool, we need to compile it with stage N-1 rustc
            (target, builder.top_stage - 1)
        }
        ToolTargetBuildMode::Dist(target_compiler) => {
            assert!(target_compiler.stage > 0);
            // If we want to dist a stage N rustc, we want to attach stage N tool to it.
            // And to build that tool, we need to compile it with stage N-1 rustc
            (target_compiler.host, target_compiler.stage - 1)
        }
    };

    let compiler = if builder.host_target == target {
        builder.compiler(build_compiler_stage, builder.host_target)
    } else {
        // If we are cross-compiling a stage 1 tool, we cannot do that with a stage 0 compiler,
        // so we auto-bump the tool's stage to 2, which means we need a stage 1 compiler.
        builder.compiler(build_compiler_stage.max(1), builder.host_target)
    };
    builder.std(compiler, target);
    compiler
}

/// Links a built tool binary with the given `name` from the build directory to the
/// tools directory.
fn copy_link_tool_bin(
    builder: &Builder<'_>,
    compiler: Compiler,
    target: TargetSelection,
    mode: Mode,
    name: &str,
) -> PathBuf {
    let cargo_out = builder.cargo_out(compiler, mode, target).join(exe(name, target));
    let bin = builder.tools_dir(compiler).join(exe(name, target));
    builder.copy_link(&cargo_out, &bin, FileType::Executable);
    bin
}

macro_rules! bootstrap_tool {
    ($(
        $name:ident, $path:expr, $tool_name:expr
        $(,is_external_tool = $external:expr)*
        $(,is_unstable_tool = $unstable:expr)*
        $(,allow_features = $allow_features:expr)?
        $(,submodules = $submodules:expr)?
        $(,artifact_kind = $artifact_kind:expr)?
        ;
    )+) => {
        #[derive(PartialEq, Eq, Clone)]
        pub enum Tool {
            $(
                $name,
            )+
        }

        impl<'a> Builder<'a> {
            pub fn tool_exe(&self, tool: Tool) -> PathBuf {
                match tool {
                    $(Tool::$name =>
                        self.ensure($name {
                            compiler: self.compiler(0, self.config.host_target),
                            target: self.config.host_target,
                        }).tool_path,
                    )+
                }
            }
        }

        $(
            #[derive(Debug, Clone, Hash, PartialEq, Eq)]
        pub struct $name {
            pub compiler: Compiler,
            pub target: TargetSelection,
        }

        impl Step for $name {
            type Output = ToolBuildResult;

            fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                run.path($path)
            }

            fn make_run(run: RunConfig<'_>) {
                run.builder.ensure($name {
                    // snapshot compiler
                    compiler: run.builder.compiler(0, run.builder.config.host_target),
                    target: run.target,
                });
            }

            #[cfg_attr(
                feature = "tracing",
                instrument(
                    level = "debug",
                    name = $tool_name,
                    skip_all,
                ),
            )]
            fn run(self, builder: &Builder<'_>) -> ToolBuildResult {
                $(
                    for submodule in $submodules {
                        builder.require_submodule(submodule, None);
                    }
                )*

                let is_unstable = false $(|| $unstable)*;
                let compiletest_wants_stage0 = $tool_name == "compiletest" && builder.config.compiletest_use_stage0_libtest;

                builder.ensure(ToolBuild {
                    build_compiler: self.compiler,
                    target: self.target,
                    tool: $tool_name,
                    mode: if is_unstable && !compiletest_wants_stage0 {
                        // use in-tree libraries for unstable features
                        Mode::ToolStd
                    } else {
                        Mode::ToolBootstrap
                    },
                    path: $path,
                    source_type: if false $(|| $external)* {
                        SourceType::Submodule
                    } else {
                        SourceType::InTree
                    },
                    extra_features: vec![],
                    allow_features: {
                        let mut _value = "";
                        $( _value = $allow_features; )?
                        _value
                    },
                    cargo_args: vec![],
                    artifact_kind: if false $(|| $artifact_kind == ToolArtifactKind::Library)* {
                        ToolArtifactKind::Library
                    } else {
                        ToolArtifactKind::Binary
                    }
                })
            }

            fn metadata(&self) -> Option<StepMetadata> {
                Some(
                    StepMetadata::build(stringify!($name), self.target)
                        .built_by(self.compiler)
                )
            }
        }
        )+
    }
}

pub(crate) const COMPILETEST_ALLOW_FEATURES: &str = "internal_output_capture";

bootstrap_tool!(
    // This is marked as an external tool because it includes dependencies
    // from submodules. Trying to keep the lints in sync between all the repos
    // is a bit of a pain. Unfortunately it means the rustbook source itself
    // doesn't deny warnings, but it is a relatively small piece of code.
    Rustbook, "src/tools/rustbook", "rustbook", is_external_tool = true, submodules = SUBMODULES_FOR_RUSTBOOK;
    UnstableBookGen, "src/tools/unstable-book-gen", "unstable-book-gen";
    Tidy, "src/tools/tidy", "tidy";
    Linkchecker, "src/tools/linkchecker", "linkchecker";
    CargoTest, "src/tools/cargotest", "cargotest";
    Compiletest, "src/tools/compiletest", "compiletest", is_unstable_tool = true, allow_features = COMPILETEST_ALLOW_FEATURES;
    BuildManifest, "src/tools/build-manifest", "build-manifest";
    RemoteTestClient, "src/tools/remote-test-client", "remote-test-client";
    RustInstaller, "src/tools/rust-installer", "rust-installer";
    RustdocTheme, "src/tools/rustdoc-themes", "rustdoc-themes";
    LintDocs, "src/tools/lint-docs", "lint-docs";
    JsonDocCk, "src/tools/jsondocck", "jsondocck";
    JsonDocLint, "src/tools/jsondoclint", "jsondoclint";
    HtmlChecker, "src/tools/html-checker", "html-checker";
    BumpStage0, "src/tools/bump-stage0", "bump-stage0";
    ReplaceVersionPlaceholder, "src/tools/replace-version-placeholder", "replace-version-placeholder";
    CollectLicenseMetadata, "src/tools/collect-license-metadata", "collect-license-metadata";
    GenerateCopyright, "src/tools/generate-copyright", "generate-copyright";
    GenerateWindowsSys, "src/tools/generate-windows-sys", "generate-windows-sys";
    // rustdoc-gui-test has a crate dependency on compiletest, so it needs the same unstable features.
    RustdocGUITest, "src/tools/rustdoc-gui-test", "rustdoc-gui-test", is_unstable_tool = true, allow_features = COMPILETEST_ALLOW_FEATURES;
    CoverageDump, "src/tools/coverage-dump", "coverage-dump";
    UnicodeTableGenerator, "src/tools/unicode-table-generator", "unicode-table-generator";
    FeaturesStatusDump, "src/tools/features-status-dump", "features-status-dump";
    OptimizedDist, "src/tools/opt-dist", "opt-dist", submodules = &["src/tools/rustc-perf"];
    RunMakeSupport, "src/tools/run-make-support", "run_make_support", artifact_kind = ToolArtifactKind::Library;
);

/// These are the submodules that are required for rustbook to work due to
/// depending on mdbook plugins.
pub static SUBMODULES_FOR_RUSTBOOK: &[&str] = &["src/doc/book", "src/doc/reference"];

/// The [rustc-perf](https://github.com/rust-lang/rustc-perf) benchmark suite, which is added
/// as a submodule at `src/tools/rustc-perf`.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RustcPerf {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for RustcPerf {
    /// Path to the built `collector` binary.
    type Output = ToolBuildResult;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/rustc-perf")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(RustcPerf {
            compiler: run.builder.compiler(0, run.builder.config.host_target),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) -> ToolBuildResult {
        // We need to ensure the rustc-perf submodule is initialized.
        builder.require_submodule("src/tools/rustc-perf", None);

        let tool = ToolBuild {
            build_compiler: self.compiler,
            target: self.target,
            tool: "collector",
            mode: Mode::ToolBootstrap,
            path: "src/tools/rustc-perf",
            source_type: SourceType::Submodule,
            extra_features: Vec::new(),
            allow_features: "",
            // Only build the collector package, which is used for benchmarking through
            // a CLI.
            cargo_args: vec!["-p".to_string(), "collector".to_string()],
            artifact_kind: ToolArtifactKind::Binary,
        };
        let res = builder.ensure(tool.clone());
        // We also need to symlink the `rustc-fake` binary to the corresponding directory,
        // because `collector` expects it in the same directory.
        copy_link_tool_bin(builder, tool.build_compiler, tool.target, tool.mode, "rustc-fake");

        res
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Ord, PartialOrd)]
pub struct ErrorIndex {
    pub compiler: Compiler,
}

impl ErrorIndex {
    pub fn command(builder: &Builder<'_>) -> BootstrapCommand {
        // Error-index-generator links with the rustdoc library, so we need to add `rustc_lib_paths`
        // for rustc_private and libLLVM.so, and `sysroot_lib` for libstd, etc.
        let host = builder.config.host_target;
        let compiler = builder.compiler_for(builder.top_stage, host, host);
        let mut cmd = command(builder.ensure(ErrorIndex { compiler }).tool_path);
        let mut dylib_paths = builder.rustc_lib_paths(compiler);
        dylib_paths.push(builder.sysroot_target_libdir(compiler, compiler.host));
        add_dylib_path(dylib_paths, &mut cmd);
        cmd
    }
}

impl Step for ErrorIndex {
    type Output = ToolBuildResult;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/error_index_generator")
    }

    fn make_run(run: RunConfig<'_>) {
        // NOTE: This `make_run` isn't used in normal situations, only if you
        // manually build the tool with `x.py build
        // src/tools/error-index-generator` which almost nobody does.
        // Normally, `x.py test` or `x.py doc` will use the
        // `ErrorIndex::command` function instead.
        let compiler = run.builder.compiler(run.builder.top_stage, run.builder.config.host_target);
        run.builder.ensure(ErrorIndex { compiler });
    }

    fn run(self, builder: &Builder<'_>) -> ToolBuildResult {
        builder.ensure(ToolBuild {
            build_compiler: self.compiler,
            target: self.compiler.host,
            tool: "error_index_generator",
            mode: Mode::ToolRustc,
            path: "src/tools/error_index_generator",
            source_type: SourceType::InTree,
            extra_features: Vec::new(),
            allow_features: "",
            cargo_args: Vec::new(),
            artifact_kind: ToolArtifactKind::Binary,
        })
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RemoteTestServer {
    pub build_compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for RemoteTestServer {
    type Output = ToolBuildResult;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/remote-test-server")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(RemoteTestServer {
            build_compiler: get_tool_target_compiler(
                run.builder,
                ToolTargetBuildMode::Build(run.target),
            ),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) -> ToolBuildResult {
        builder.ensure(ToolBuild {
            build_compiler: self.build_compiler,
            target: self.target,
            tool: "remote-test-server",
            mode: Mode::ToolTarget,
            path: "src/tools/remote-test-server",
            source_type: SourceType::InTree,
            extra_features: Vec::new(),
            allow_features: "",
            cargo_args: Vec::new(),
            artifact_kind: ToolArtifactKind::Binary,
        })
    }

    fn metadata(&self) -> Option<StepMetadata> {
        Some(StepMetadata::build("remote-test-server", self.target).built_by(self.build_compiler))
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Ord, PartialOrd)]
pub struct Rustdoc {
    /// This should only ever be 0 or 2.
    /// We sometimes want to reference the "bootstrap" rustdoc, which is why this option is here.
    pub compiler: Compiler,
}

impl Step for Rustdoc {
    type Output = ToolBuildResult;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/rustdoc").path("src/librustdoc")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder
            .ensure(Rustdoc { compiler: run.builder.compiler(run.builder.top_stage, run.target) });
    }

    fn run(self, builder: &Builder<'_>) -> ToolBuildResult {
        let target_compiler = self.compiler;
        let target = target_compiler.host;

        if target_compiler.stage == 0 {
            if !target_compiler.is_snapshot(builder) {
                panic!("rustdoc in stage 0 must be snapshot rustdoc");
            }

            return ToolBuildResult {
                tool_path: builder.initial_rustdoc.clone(),
                build_compiler: target_compiler,
                target_compiler,
            };
        }

        let bin_rustdoc = || {
            let sysroot = builder.sysroot(target_compiler);
            let bindir = sysroot.join("bin");
            t!(fs::create_dir_all(&bindir));
            let bin_rustdoc = bindir.join(exe("rustdoc", target_compiler.host));
            let _ = fs::remove_file(&bin_rustdoc);
            bin_rustdoc
        };

        // If CI rustc is enabled and we haven't modified the rustdoc sources,
        // use the precompiled rustdoc from CI rustc's sysroot to speed up bootstrapping.
        if builder.download_rustc()
            && target_compiler.stage > 0
            && builder.rust_info().is_managed_git_subrepository()
        {
            let files_to_track = &["src/librustdoc", "src/tools/rustdoc", "src/rustdoc-json-types"];

            // Check if unchanged
            if !builder.config.has_changes_from_upstream(files_to_track) {
                let precompiled_rustdoc = builder
                    .config
                    .ci_rustc_dir()
                    .join("bin")
                    .join(exe("rustdoc", target_compiler.host));

                let bin_rustdoc = bin_rustdoc();
                builder.copy_link(&precompiled_rustdoc, &bin_rustdoc, FileType::Executable);

                return ToolBuildResult {
                    tool_path: bin_rustdoc,
                    build_compiler: target_compiler,
                    target_compiler,
                };
            }
        }

        // The presence of `target_compiler` ensures that the necessary libraries (codegen backends,
        // compiler libraries, ...) are built. Rustdoc does not require the presence of any
        // libraries within sysroot_libdir (i.e., rustlib), though doctests may want it (since
        // they'll be linked to those libraries). As such, don't explicitly `ensure` any additional
        // libraries here. The intuition here is that If we've built a compiler, we should be able
        // to build rustdoc.
        //
        let mut extra_features = Vec::new();
        if builder.config.jemalloc(target) {
            extra_features.push("jemalloc".to_string());
        }

        let ToolBuildResult { tool_path, build_compiler, target_compiler } =
            builder.ensure(ToolBuild {
                build_compiler: target_compiler,
                target,
                // Cargo adds a number of paths to the dylib search path on windows, which results in
                // the wrong rustdoc being executed. To avoid the conflicting rustdocs, we name the "tool"
                // rustdoc a different name.
                tool: "rustdoc_tool_binary",
                mode: Mode::ToolRustc,
                path: "src/tools/rustdoc",
                source_type: SourceType::InTree,
                extra_features,
                allow_features: "",
                cargo_args: Vec::new(),
                artifact_kind: ToolArtifactKind::Binary,
            });

        // don't create a stage0-sysroot/bin directory.
        if target_compiler.stage > 0 {
            if builder.config.rust_debuginfo_level_tools == DebuginfoLevel::None {
                // Due to LTO a lot of debug info from C++ dependencies such as jemalloc can make it into
                // our final binaries
                compile::strip_debug(builder, target, &tool_path);
            }
            let bin_rustdoc = bin_rustdoc();
            builder.copy_link(&tool_path, &bin_rustdoc, FileType::Executable);
            ToolBuildResult { tool_path: bin_rustdoc, build_compiler, target_compiler }
        } else {
            ToolBuildResult { tool_path, build_compiler, target_compiler }
        }
    }

    fn metadata(&self) -> Option<StepMetadata> {
        Some(
            StepMetadata::build("rustdoc", self.compiler.host)
                // rustdoc is ToolRustc, so stage N rustdoc is built by stage N-1 rustc
                // FIXME: make this stage deduction automatic somehow
                // FIXME: log the compiler that actually built ToolRustc steps
                .stage(self.compiler.stage.saturating_sub(1)),
        )
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Cargo {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for Cargo {
    type Output = ToolBuildResult;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.path("src/tools/cargo").default_condition(builder.tool_enabled("cargo"))
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Cargo {
            compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.host_target),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) -> ToolBuildResult {
        builder.build.require_submodule("src/tools/cargo", None);

        builder.ensure(ToolBuild {
            build_compiler: self.compiler,
            target: self.target,
            tool: "cargo",
            mode: Mode::ToolRustc,
            path: "src/tools/cargo",
            source_type: SourceType::Submodule,
            extra_features: Vec::new(),
            allow_features: "",
            cargo_args: Vec::new(),
            artifact_kind: ToolArtifactKind::Binary,
        })
    }
}

/// Represents a built LldWrapper, the `lld-wrapper` tool itself, and a directory
/// containing a build of LLD.
#[derive(Clone)]
pub struct BuiltLldWrapper {
    tool: ToolBuildResult,
    lld_dir: PathBuf,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct LldWrapper {
    pub build_compiler: Compiler,
    pub target: TargetSelection,
}

impl LldWrapper {
    /// Returns `LldWrapper` that should be **used** by the passed compiler.
    pub fn for_use_by_compiler(builder: &Builder<'_>, target_compiler: Compiler) -> Self {
        Self {
            build_compiler: get_tool_target_compiler(
                builder,
                ToolTargetBuildMode::Dist(target_compiler),
            ),
            target: target_compiler.host,
        }
    }
}

impl Step for LldWrapper {
    type Output = BuiltLldWrapper;

    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/lld-wrapper")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(LldWrapper {
            build_compiler: get_tool_target_compiler(
                run.builder,
                ToolTargetBuildMode::Build(run.target),
            ),
            target: run.target,
        });
    }

    #[cfg_attr(
        feature = "tracing",
        instrument(
            level = "debug",
            name = "LldWrapper::run",
            skip_all,
            fields(build_compiler = ?self.build_compiler),
        ),
    )]
    fn run(self, builder: &Builder<'_>) -> Self::Output {
        let lld_dir = builder.ensure(llvm::Lld { target: self.target });
        let tool = builder.ensure(ToolBuild {
            build_compiler: self.build_compiler,
            target: self.target,
            tool: "lld-wrapper",
            mode: Mode::ToolTarget,
            path: "src/tools/lld-wrapper",
            source_type: SourceType::InTree,
            extra_features: Vec::new(),
            allow_features: "",
            cargo_args: Vec::new(),
            artifact_kind: ToolArtifactKind::Binary,
        });
        BuiltLldWrapper { tool, lld_dir }
    }

    fn metadata(&self) -> Option<StepMetadata> {
        Some(StepMetadata::build("LldWrapper", self.target).built_by(self.build_compiler))
    }
}

pub(crate) fn copy_lld_artifacts(
    builder: &Builder<'_>,
    lld_wrapper: BuiltLldWrapper,
    target_compiler: Compiler,
) {
    let target = target_compiler.host;

    let libdir_bin = builder.sysroot_target_bindir(target_compiler, target);
    t!(fs::create_dir_all(&libdir_bin));

    let src_exe = exe("lld", target);
    let dst_exe = exe("rust-lld", target);

    builder.copy_link(
        &lld_wrapper.lld_dir.join("bin").join(src_exe),
        &libdir_bin.join(dst_exe),
        FileType::Executable,
    );
    let self_contained_lld_dir = libdir_bin.join("gcc-ld");
    t!(fs::create_dir_all(&self_contained_lld_dir));

    for name in crate::LLD_FILE_NAMES {
        builder.copy_link(
            &lld_wrapper.tool.tool_path,
            &self_contained_lld_dir.join(exe(name, target)),
            FileType::Executable,
        );
    }
}

/// Builds the `wasm-component-ld` linker wrapper, which is shipped with rustc to be executed on the
/// host platform where rustc runs.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct WasmComponentLd {
    build_compiler: Compiler,
    target: TargetSelection,
}

impl WasmComponentLd {
    /// Returns `WasmComponentLd` that should be **used** by the passed compiler.
    pub fn for_use_by_compiler(builder: &Builder<'_>, target_compiler: Compiler) -> Self {
        Self {
            build_compiler: get_tool_target_compiler(
                builder,
                ToolTargetBuildMode::Dist(target_compiler),
            ),
            target: target_compiler.host,
        }
    }
}

impl Step for WasmComponentLd {
    type Output = ToolBuildResult;

    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/wasm-component-ld")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(WasmComponentLd {
            build_compiler: get_tool_target_compiler(
                run.builder,
                ToolTargetBuildMode::Build(run.target),
            ),
            target: run.target,
        });
    }

    #[cfg_attr(
        feature = "tracing",
        instrument(
            level = "debug",
            name = "WasmComponentLd::run",
            skip_all,
            fields(build_compiler = ?self.build_compiler),
        ),
    )]
    fn run(self, builder: &Builder<'_>) -> ToolBuildResult {
        builder.ensure(ToolBuild {
            build_compiler: self.build_compiler,
            target: self.target,
            tool: "wasm-component-ld",
            mode: Mode::ToolTarget,
            path: "src/tools/wasm-component-ld",
            source_type: SourceType::InTree,
            extra_features: vec![],
            allow_features: "",
            cargo_args: vec![],
            artifact_kind: ToolArtifactKind::Binary,
        })
    }

    fn metadata(&self) -> Option<StepMetadata> {
        Some(StepMetadata::build("WasmComponentLd", self.target).built_by(self.build_compiler))
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RustAnalyzer {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl RustAnalyzer {
    pub const ALLOW_FEATURES: &'static str = "rustc_private,proc_macro_internals,proc_macro_diagnostic,proc_macro_span,proc_macro_span_shrink,proc_macro_def_site";
}

impl Step for RustAnalyzer {
    type Output = ToolBuildResult;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.path("src/tools/rust-analyzer").default_condition(builder.tool_enabled("rust-analyzer"))
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(RustAnalyzer {
            compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.host_target),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) -> ToolBuildResult {
        builder.ensure(ToolBuild {
            build_compiler: self.compiler,
            target: self.target,
            tool: "rust-analyzer",
            mode: Mode::ToolRustc,
            path: "src/tools/rust-analyzer",
            extra_features: vec!["in-rust-tree".to_owned()],
            source_type: SourceType::InTree,
            allow_features: RustAnalyzer::ALLOW_FEATURES,
            cargo_args: Vec::new(),
            artifact_kind: ToolArtifactKind::Binary,
        })
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RustAnalyzerProcMacroSrv {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for RustAnalyzerProcMacroSrv {
    type Output = Option<ToolBuildResult>;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        // Allow building `rust-analyzer-proc-macro-srv` both as part of the `rust-analyzer` and as a stand-alone tool.
        run.path("src/tools/rust-analyzer")
            .path("src/tools/rust-analyzer/crates/proc-macro-srv-cli")
            .default_condition(
                builder.tool_enabled("rust-analyzer")
                    || builder.tool_enabled("rust-analyzer-proc-macro-srv"),
            )
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(RustAnalyzerProcMacroSrv {
            compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.host_target),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) -> Option<ToolBuildResult> {
        let tool_result = builder.ensure(ToolBuild {
            build_compiler: self.compiler,
            target: self.target,
            tool: "rust-analyzer-proc-macro-srv",
            mode: Mode::ToolRustc,
            path: "src/tools/rust-analyzer/crates/proc-macro-srv-cli",
            extra_features: vec!["in-rust-tree".to_owned()],
            source_type: SourceType::InTree,
            allow_features: RustAnalyzer::ALLOW_FEATURES,
            cargo_args: Vec::new(),
            artifact_kind: ToolArtifactKind::Binary,
        });

        // Copy `rust-analyzer-proc-macro-srv` to `<sysroot>/libexec/`
        // so that r-a can use it.
        let libexec_path = builder.sysroot(self.compiler).join("libexec");
        t!(fs::create_dir_all(&libexec_path));
        builder.copy_link(
            &tool_result.tool_path,
            &libexec_path.join("rust-analyzer-proc-macro-srv"),
            FileType::Executable,
        );

        Some(tool_result)
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct LlvmBitcodeLinker {
    build_compiler: Compiler,
    target: TargetSelection,
}

impl LlvmBitcodeLinker {
    /// Returns `LlvmBitcodeLinker` that will be **compiled** by the passed compiler, for the given
    /// `target`.
    pub fn from_build_compiler(build_compiler: Compiler, target: TargetSelection) -> Self {
        Self { build_compiler, target }
    }

    /// Returns `LlvmBitcodeLinker` that should be **used** by the passed compiler.
    pub fn from_target_compiler(builder: &Builder<'_>, target_compiler: Compiler) -> Self {
        Self {
            build_compiler: get_tool_target_compiler(
                builder,
                ToolTargetBuildMode::Dist(target_compiler),
            ),
            target: target_compiler.host,
        }
    }

    /// Return a compiler that is able to build this tool for the given `target`.
    pub fn get_build_compiler_for_target(
        builder: &Builder<'_>,
        target: TargetSelection,
    ) -> Compiler {
        get_tool_target_compiler(builder, ToolTargetBuildMode::Build(target))
    }
}

impl Step for LlvmBitcodeLinker {
    type Output = ToolBuildResult;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.path("src/tools/llvm-bitcode-linker")
            .default_condition(builder.tool_enabled("llvm-bitcode-linker"))
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(LlvmBitcodeLinker {
            build_compiler: Self::get_build_compiler_for_target(run.builder, run.target),
            target: run.target,
        });
    }

    #[cfg_attr(
        feature = "tracing",
        instrument(level = "debug", name = "LlvmBitcodeLinker::run", skip_all)
    )]
    fn run(self, builder: &Builder<'_>) -> ToolBuildResult {
        builder.ensure(ToolBuild {
            build_compiler: self.build_compiler,
            target: self.target,
            tool: "llvm-bitcode-linker",
            mode: Mode::ToolTarget,
            path: "src/tools/llvm-bitcode-linker",
            source_type: SourceType::InTree,
            extra_features: vec![],
            allow_features: "",
            cargo_args: Vec::new(),
            artifact_kind: ToolArtifactKind::Binary,
        })
    }

    fn metadata(&self) -> Option<StepMetadata> {
        Some(StepMetadata::build("LlvmBitcodeLinker", self.target).built_by(self.build_compiler))
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct LibcxxVersionTool {
    pub target: TargetSelection,
}

#[expect(dead_code)]
#[derive(Debug, Clone)]
pub enum LibcxxVersion {
    Gnu(usize),
    Llvm(usize),
}

impl Step for LibcxxVersionTool {
    type Output = LibcxxVersion;
    const DEFAULT: bool = false;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    fn run(self, builder: &Builder<'_>) -> LibcxxVersion {
        let out_dir = builder.out.join(self.target.to_string()).join("libcxx-version");
        let executable = out_dir.join(exe("libcxx-version", self.target));

        // This is a sanity-check specific step, which means it is frequently called (when using
        // CI LLVM), and compiling `src/tools/libcxx-version/main.cpp` at the beginning of the bootstrap
        // invocation adds a fair amount of overhead to the process (see https://github.com/rust-lang/rust/issues/126423).
        // Therefore, we want to avoid recompiling this file unnecessarily.
        if !executable.exists() {
            if !out_dir.exists() {
                t!(fs::create_dir_all(&out_dir));
            }

            let compiler = builder.cxx(self.target).unwrap();
            let mut cmd = command(compiler);

            cmd.arg("-o")
                .arg(&executable)
                .arg(builder.src.join("src/tools/libcxx-version/main.cpp"));

            cmd.run(builder);

            if !executable.exists() {
                panic!("Something went wrong. {} is not present", executable.display());
            }
        }

        let version_output = command(executable).run_capture_stdout(builder).stdout();

        let version_str = version_output.split_once("version:").unwrap().1;
        let version = version_str.trim().parse::<usize>().unwrap();

        if version_output.starts_with("libstdc++") {
            LibcxxVersion::Gnu(version)
        } else if version_output.starts_with("libc++") {
            LibcxxVersion::Llvm(version)
        } else {
            panic!("Coudln't recognize the standard library version.");
        }
    }
}

macro_rules! tool_extended {
    (
        $name:ident {
            path: $path:expr,
            tool_name: $tool_name:expr,
            stable: $stable:expr
            $( , add_bins_to_sysroot: $add_bins_to_sysroot:expr )?
            $( , add_features: $add_features:expr )?
            $( , cargo_args: $cargo_args:expr )?
            $( , )?
        }
    ) => {
        #[derive(Debug, Clone, Hash, PartialEq, Eq)]
        pub struct $name {
            pub compiler: Compiler,
            pub target: TargetSelection,
        }

        impl Step for $name {
            type Output = ToolBuildResult;
            const DEFAULT: bool = true; // Overridden by `should_run_tool_build_step`
            const ONLY_HOSTS: bool = true;

            fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                should_run_tool_build_step(
                    run,
                    $tool_name,
                    $path,
                    $stable,
                )
            }

            fn make_run(run: RunConfig<'_>) {
                run.builder.ensure($name {
                    compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.host_target),
                    target: run.target,
                });
            }

            fn run(self, builder: &Builder<'_>) -> ToolBuildResult {
                let Self { compiler, target } = self;
                run_tool_build_step(
                    builder,
                    compiler,
                    target,
                    $tool_name,
                    $path,
                    None $( .or(Some(&$add_bins_to_sysroot)) )?,
                    None $( .or(Some($add_features)) )?,
                    None $( .or(Some($cargo_args)) )?,
                )
            }

            fn metadata(&self) -> Option<StepMetadata> {
                // FIXME: refactor extended tool steps to make the build_compiler explicit,
                // it is offset by one now for rustc tools
                Some(
                    StepMetadata::build($tool_name, self.target)
                        .built_by(self.compiler.with_stage(self.compiler.stage.saturating_sub(1)))
                )
            }
        }
    }
}

fn should_run_tool_build_step<'a>(
    run: ShouldRun<'a>,
    tool_name: &'static str,
    path: &'static str,
    stable: bool,
) -> ShouldRun<'a> {
    let builder = run.builder;
    run.path(path).default_condition(
        builder.config.extended
            && builder.config.tools.as_ref().map_or(
                // By default, on nightly/dev enable all tools, else only
                // build stable tools.
                stable || builder.build.unstable_features(),
                // If `tools` is set, search list for this tool.
                |tools| {
                    tools.iter().any(|tool| match tool.as_ref() {
                        "clippy" => tool_name == "clippy-driver",
                        x => tool_name == x,
                    })
                },
            ),
    )
}

#[expect(clippy::too_many_arguments)] // silence overeager clippy lint
fn run_tool_build_step(
    builder: &Builder<'_>,
    compiler: Compiler,
    target: TargetSelection,
    tool_name: &'static str,
    path: &'static str,
    add_bins_to_sysroot: Option<&[&str]>,
    add_features: Option<fn(&Builder<'_>, TargetSelection, &mut Vec<String>)>,
    cargo_args: Option<&[&'static str]>,
) -> ToolBuildResult {
    let mut extra_features = Vec::new();
    if let Some(func) = add_features {
        func(builder, target, &mut extra_features);
    }

    let ToolBuildResult { tool_path, build_compiler, target_compiler } =
        builder.ensure(ToolBuild {
            build_compiler: compiler,
            target,
            tool: tool_name,
            mode: Mode::ToolRustc,
            path,
            extra_features,
            source_type: SourceType::InTree,
            allow_features: "",
            cargo_args: cargo_args.unwrap_or_default().iter().map(|s| String::from(*s)).collect(),
            artifact_kind: ToolArtifactKind::Binary,
        });

    if let Some(add_bins_to_sysroot) = add_bins_to_sysroot
        && !add_bins_to_sysroot.is_empty()
        && target_compiler.stage > 0
    {
        let bindir = builder.sysroot(target_compiler).join("bin");
        t!(fs::create_dir_all(&bindir));

        for add_bin in add_bins_to_sysroot {
            let bin_destination = bindir.join(exe(add_bin, target_compiler.host));
            builder.copy_link(&tool_path, &bin_destination, FileType::Executable);
        }

        // Return a path into the bin dir.
        let path = bindir.join(exe(tool_name, target_compiler.host));
        ToolBuildResult { tool_path: path, build_compiler, target_compiler }
    } else {
        ToolBuildResult { tool_path, build_compiler, target_compiler }
    }
}

tool_extended!(Cargofmt {
    path: "src/tools/rustfmt",
    tool_name: "cargo-fmt",
    stable: true,
    add_bins_to_sysroot: ["cargo-fmt"]
});
tool_extended!(CargoClippy {
    path: "src/tools/clippy",
    tool_name: "cargo-clippy",
    stable: true,
    add_bins_to_sysroot: ["cargo-clippy"]
});
tool_extended!(Clippy {
    path: "src/tools/clippy",
    tool_name: "clippy-driver",
    stable: true,
    add_bins_to_sysroot: ["clippy-driver"],
    add_features: |builder, target, features| {
        if builder.config.jemalloc(target) {
            features.push("jemalloc".to_string());
        }
    }
});
tool_extended!(Miri {
    path: "src/tools/miri",
    tool_name: "miri",
    stable: false,
    add_bins_to_sysroot: ["miri"],
    // Always compile also tests when building miri. Otherwise feature unification can cause rebuilds between building and testing miri.
    cargo_args: &["--all-targets"],
});
tool_extended!(CargoMiri {
    path: "src/tools/miri/cargo-miri",
    tool_name: "cargo-miri",
    stable: false,
    add_bins_to_sysroot: ["cargo-miri"]
});
tool_extended!(Rustfmt {
    path: "src/tools/rustfmt",
    tool_name: "rustfmt",
    stable: true,
    add_bins_to_sysroot: ["rustfmt"]
});

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TestFloatParse {
    pub host: TargetSelection,
}

impl TestFloatParse {
    pub const ALLOW_FEATURES: &'static str = "f16,cfg_target_has_reliable_f16_f128";
}

impl Step for TestFloatParse {
    type Output = ToolBuildResult;
    const ONLY_HOSTS: bool = true;
    const DEFAULT: bool = false;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/test-float-parse")
    }

    fn run(self, builder: &Builder<'_>) -> ToolBuildResult {
        let bootstrap_host = builder.config.host_target;
        let compiler = builder.compiler(builder.top_stage, bootstrap_host);

        builder.ensure(ToolBuild {
            build_compiler: compiler,
            target: bootstrap_host,
            tool: "test-float-parse",
            mode: Mode::ToolStd,
            path: "src/tools/test-float-parse",
            source_type: SourceType::InTree,
            extra_features: Vec::new(),
            allow_features: Self::ALLOW_FEATURES,
            cargo_args: Vec::new(),
            artifact_kind: ToolArtifactKind::Binary,
        })
    }
}

impl Builder<'_> {
    /// Gets a `BootstrapCommand` which is ready to run `tool` in `stage` built for
    /// `host`.
    pub fn tool_cmd(&self, tool: Tool) -> BootstrapCommand {
        let mut cmd = command(self.tool_exe(tool));
        let compiler = self.compiler(0, self.config.host_target);
        let host = &compiler.host;
        // Prepares the `cmd` provided to be able to run the `compiler` provided.
        //
        // Notably this munges the dynamic library lookup path to point to the
        // right location to run `compiler`.
        let mut lib_paths: Vec<PathBuf> =
            vec![self.cargo_out(compiler, Mode::ToolBootstrap, *host).join("deps")];

        // On MSVC a tool may invoke a C compiler (e.g., compiletest in run-make
        // mode) and that C compiler may need some extra PATH modification. Do
        // so here.
        if compiler.host.is_msvc() {
            let curpaths = env::var_os("PATH").unwrap_or_default();
            let curpaths = env::split_paths(&curpaths).collect::<Vec<_>>();
            for (k, v) in self.cc[&compiler.host].env() {
                if k != "PATH" {
                    continue;
                }
                for path in env::split_paths(v) {
                    if !curpaths.contains(&path) {
                        lib_paths.push(path);
                    }
                }
            }
        }

        add_dylib_path(lib_paths, &mut cmd);

        // Provide a RUSTC for this command to use.
        cmd.env("RUSTC", &self.initial_rustc);

        cmd
    }
}
