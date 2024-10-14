use std::path::PathBuf;
use std::{env, fs};

use build_helper::git::get_closest_merge_commit;

use crate::core::build_steps::compile;
use crate::core::build_steps::toolstate::ToolState;
use crate::core::builder;
use crate::core::builder::{Builder, Cargo as CargoCommand, RunConfig, ShouldRun, Step};
use crate::core::config::TargetSelection;
use crate::utils::channel::GitInfo;
use crate::utils::exec::{BootstrapCommand, command};
use crate::utils::helpers::{add_dylib_path, exe, git, t};
use crate::{Compiler, Kind, Mode, gha};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum SourceType {
    InTree,
    Submodule,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct ToolBuild {
    compiler: Compiler,
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
            _ => self.msg(Kind::Build, build_stage, format_args!("tool {tool}"), *host, *target),
        }
    }
}

impl Step for ToolBuild {
    type Output = PathBuf;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    /// Builds a tool in `src/tools`
    ///
    /// This will build the specified tool with the specified `host` compiler in
    /// `stage` into the normal cargo output directory.
    fn run(self, builder: &Builder<'_>) -> PathBuf {
        let compiler = self.compiler;
        let target = self.target;
        let mut tool = self.tool;
        let path = self.path;

        match self.mode {
            Mode::ToolRustc => {
                builder.ensure(compile::Std::new(compiler, compiler.host));
                builder.ensure(compile::Rustc::new(compiler, target));
            }
            Mode::ToolStd => builder.ensure(compile::Std::new(compiler, target)),
            Mode::ToolBootstrap => {} // uses downloaded stage0 compiler libs
            _ => panic!("unexpected Mode for tool build"),
        }

        let mut cargo = prepare_tool_cargo(
            builder,
            compiler,
            self.mode,
            target,
            Kind::Build,
            path,
            self.source_type,
            &self.extra_features,
        );
        if !self.allow_features.is_empty() {
            cargo.allow_features(self.allow_features);
        }
        cargo.args(self.cargo_args);
        let _guard = builder.msg_tool(
            Kind::Build,
            self.mode,
            self.tool,
            self.compiler.stage,
            &self.compiler.host,
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
            copy_link_tool_bin(builder, self.compiler, self.target, self.mode, tool)
        }
    }
}

#[allow(clippy::too_many_arguments)] // FIXME: reduce the number of args and remove this.
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

    let dir = builder.src.join(path);
    cargo.arg("--manifest-path").arg(dir.join("Cargo.toml"));

    let mut features = extra_features.to_vec();
    if builder.build.config.cargo_native_static {
        if path.ends_with("cargo")
            || path.ends_with("rls")
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

    let info = GitInfo::new(builder.config.omit_git_hash, &dir);
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
    builder.copy_link(&cargo_out, &bin);
    bin
}

macro_rules! bootstrap_tool {
    ($(
        $name:ident, $path:expr, $tool_name:expr
        $(,is_external_tool = $external:expr)*
        $(,is_unstable_tool = $unstable:expr)*
        $(,allow_features = $allow_features:expr)?
        $(,submodules = $submodules:expr)?
        ;
    )+) => {
        #[derive(PartialEq, Eq, Clone)]
        #[allow(dead_code)]
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
                            compiler: self.compiler(0, self.config.build),
                            target: self.config.build,
                        }),
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
            type Output = PathBuf;

            fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                run.path($path)
            }

            fn make_run(run: RunConfig<'_>) {
                run.builder.ensure($name {
                    // snapshot compiler
                    compiler: run.builder.compiler(0, run.builder.config.build),
                    target: run.target,
                });
            }

            fn run(self, builder: &Builder<'_>) -> PathBuf {
                $(
                    for submodule in $submodules {
                        builder.require_submodule(submodule, None);
                    }
                )*
                builder.ensure(ToolBuild {
                    compiler: self.compiler,
                    target: self.target,
                    tool: $tool_name,
                    mode: if false $(|| $unstable)* {
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
                    allow_features: concat!($($allow_features)*),
                    cargo_args: vec![]
                })
            }
        }
        )+
    }
}

bootstrap_tool!(
    Rustbook, "src/tools/rustbook", "rustbook", submodules = SUBMODULES_FOR_RUSTBOOK;
    UnstableBookGen, "src/tools/unstable-book-gen", "unstable-book-gen";
    Tidy, "src/tools/tidy", "tidy";
    Linkchecker, "src/tools/linkchecker", "linkchecker";
    CargoTest, "src/tools/cargotest", "cargotest";
    Compiletest, "src/tools/compiletest", "compiletest", is_unstable_tool = true, allow_features = "test";
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
    SuggestTests, "src/tools/suggest-tests", "suggest-tests";
    GenerateWindowsSys, "src/tools/generate-windows-sys", "generate-windows-sys";
    RustdocGUITest, "src/tools/rustdoc-gui-test", "rustdoc-gui-test", is_unstable_tool = true, allow_features = "test";
    CoverageDump, "src/tools/coverage-dump", "coverage-dump";
    RustcPerfWrapper, "src/tools/rustc-perf-wrapper", "rustc-perf-wrapper";
    WasmComponentLd, "src/tools/wasm-component-ld", "wasm-component-ld", is_unstable_tool = true, allow_features = "min_specialization";
);

/// These are the submodules that are required for rustbook to work due to
/// depending on mdbook plugins.
pub static SUBMODULES_FOR_RUSTBOOK: &[&str] = &["src/doc/book", "src/doc/reference"];

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct OptimizedDist {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for OptimizedDist {
    type Output = PathBuf;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/opt-dist")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(OptimizedDist {
            compiler: run.builder.compiler(0, run.builder.config.build),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) -> PathBuf {
        // We need to ensure the rustc-perf submodule is initialized when building opt-dist since
        // the tool requires it to be in place to run.
        builder.require_submodule("src/tools/rustc-perf", None);

        builder.ensure(ToolBuild {
            compiler: self.compiler,
            target: self.target,
            tool: "opt-dist",
            mode: Mode::ToolBootstrap,
            path: "src/tools/opt-dist",
            source_type: SourceType::InTree,
            extra_features: Vec::new(),
            allow_features: "",
            cargo_args: Vec::new(),
        })
    }
}

/// The [rustc-perf](https://github.com/rust-lang/rustc-perf) benchmark suite, which is added
/// as a submodule at `src/tools/rustc-perf`.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RustcPerf {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for RustcPerf {
    /// Path to the built `collector` binary.
    type Output = PathBuf;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/rustc-perf")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(RustcPerf {
            compiler: run.builder.compiler(0, run.builder.config.build),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) -> PathBuf {
        // We need to ensure the rustc-perf submodule is initialized.
        builder.require_submodule("src/tools/rustc-perf", None);

        let tool = ToolBuild {
            compiler: self.compiler,
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
        };
        let collector_bin = builder.ensure(tool.clone());
        // We also need to symlink the `rustc-fake` binary to the corresponding directory,
        // because `collector` expects it in the same directory.
        copy_link_tool_bin(builder, tool.compiler, tool.target, tool.mode, "rustc-fake");

        collector_bin
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
        let host = builder.config.build;
        let compiler = builder.compiler_for(builder.top_stage, host, host);
        let mut cmd = command(builder.ensure(ErrorIndex { compiler }));
        let mut dylib_paths = builder.rustc_lib_paths(compiler);
        dylib_paths.push(PathBuf::from(&builder.sysroot_libdir(compiler, compiler.host)));
        add_dylib_path(dylib_paths, &mut cmd);
        cmd
    }
}

impl Step for ErrorIndex {
    type Output = PathBuf;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/error_index_generator")
    }

    fn make_run(run: RunConfig<'_>) {
        // Compile the error-index in the same stage as rustdoc to avoid
        // recompiling rustdoc twice if we can.
        //
        // NOTE: This `make_run` isn't used in normal situations, only if you
        // manually build the tool with `x.py build
        // src/tools/error-index-generator` which almost nobody does.
        // Normally, `x.py test` or `x.py doc` will use the
        // `ErrorIndex::command` function instead.
        let compiler =
            run.builder.compiler(run.builder.top_stage.saturating_sub(1), run.builder.config.build);
        run.builder.ensure(ErrorIndex { compiler });
    }

    fn run(self, builder: &Builder<'_>) -> PathBuf {
        builder.ensure(ToolBuild {
            compiler: self.compiler,
            target: self.compiler.host,
            tool: "error_index_generator",
            mode: Mode::ToolRustc,
            path: "src/tools/error_index_generator",
            source_type: SourceType::InTree,
            extra_features: Vec::new(),
            allow_features: "",
            cargo_args: Vec::new(),
        })
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RemoteTestServer {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for RemoteTestServer {
    type Output = PathBuf;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/remote-test-server")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(RemoteTestServer {
            compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.build),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) -> PathBuf {
        builder.ensure(ToolBuild {
            compiler: self.compiler,
            target: self.target,
            tool: "remote-test-server",
            mode: Mode::ToolStd,
            path: "src/tools/remote-test-server",
            source_type: SourceType::InTree,
            extra_features: Vec::new(),
            allow_features: "",
            cargo_args: Vec::new(),
        })
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Ord, PartialOrd)]
pub struct Rustdoc {
    /// This should only ever be 0 or 2.
    /// We sometimes want to reference the "bootstrap" rustdoc, which is why this option is here.
    pub compiler: Compiler,
}

impl Step for Rustdoc {
    type Output = PathBuf;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.path("src/tools/rustdoc").path("src/librustdoc")
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Rustdoc {
            // NOTE: this is somewhat unique in that we actually want a *target*
            // compiler here, because rustdoc *is* a compiler. We won't be using
            // this as the compiler to build with, but rather this is "what
            // compiler are we producing"?
            compiler: run.builder.compiler(run.builder.top_stage, run.target),
        });
    }

    fn run(self, builder: &Builder<'_>) -> PathBuf {
        let target_compiler = self.compiler;
        if target_compiler.stage == 0 {
            if !target_compiler.is_snapshot(builder) {
                panic!("rustdoc in stage 0 must be snapshot rustdoc");
            }
            return builder.initial_rustc.with_file_name(exe("rustdoc", target_compiler.host));
        }
        let target = target_compiler.host;

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
            let commit = get_closest_merge_commit(
                Some(&builder.config.src),
                &builder.config.git_config(),
                &[],
            )
            .unwrap();

            let librustdoc_src = builder.config.src.join("src/librustdoc");
            let rustdoc_src = builder.config.src.join("src/tools/rustdoc");

            // FIXME: The change detection logic here is quite similar to `Config::download_ci_rustc_commit`.
            // It would be better to unify them.
            let has_changes = !git(Some(&builder.config.src))
                .allow_failure()
                .run_always()
                .args(["diff-index", "--quiet", &commit])
                .arg("--")
                .arg(librustdoc_src)
                .arg(rustdoc_src)
                .run(builder);

            if !has_changes {
                let precompiled_rustdoc = builder
                    .config
                    .ci_rustc_dir()
                    .join("bin")
                    .join(exe("rustdoc", target_compiler.host));

                let bin_rustdoc = bin_rustdoc();
                builder.copy_link(&precompiled_rustdoc, &bin_rustdoc);
                return bin_rustdoc;
            }
        }

        let build_compiler = if builder.download_rustc() && target_compiler.stage == 1 {
            // We already have the stage 1 compiler, we don't need to cut the stage.
            builder.compiler(target_compiler.stage, builder.config.build)
        } else {
            // Similar to `compile::Assemble`, build with the previous stage's compiler. Otherwise
            // we'd have stageN/bin/rustc and stageN/bin/rustdoc be effectively different stage
            // compilers, which isn't what we want. Rustdoc should be linked in the same way as the
            // rustc compiler it's paired with, so it must be built with the previous stage compiler.
            builder.compiler(target_compiler.stage - 1, builder.config.build)
        };

        // When using `download-rustc` and a stage0 build_compiler, copying rustc doesn't actually
        // build stage0 libstd (because the libstd in sysroot has the wrong ABI). Explicitly build
        // it.
        builder.ensure(compile::Std::new(build_compiler, target_compiler.host));
        builder.ensure(compile::Rustc::new(build_compiler, target_compiler.host));

        // The presence of `target_compiler` ensures that the necessary libraries (codegen backends,
        // compiler libraries, ...) are built. Rustdoc does not require the presence of any
        // libraries within sysroot_libdir (i.e., rustlib), though doctests may want it (since
        // they'll be linked to those libraries). As such, don't explicitly `ensure` any additional
        // libraries here. The intuition here is that If we've built a compiler, we should be able
        // to build rustdoc.
        //
        let mut features = Vec::new();
        if builder.config.jemalloc {
            features.push("jemalloc".to_string());
        }

        // NOTE: Never modify the rustflags here, it breaks the build cache for other tools!
        let cargo = prepare_tool_cargo(
            builder,
            build_compiler,
            Mode::ToolRustc,
            target,
            Kind::Build,
            "src/tools/rustdoc",
            SourceType::InTree,
            features.as_slice(),
        );

        let _guard = builder.msg_tool(
            Kind::Build,
            Mode::ToolRustc,
            "rustdoc",
            build_compiler.stage,
            &self.compiler.host,
            &target,
        );
        cargo.into_cmd().run(builder);

        // Cargo adds a number of paths to the dylib search path on windows, which results in
        // the wrong rustdoc being executed. To avoid the conflicting rustdocs, we name the "tool"
        // rustdoc a different name.
        let tool_rustdoc = builder
            .cargo_out(build_compiler, Mode::ToolRustc, target)
            .join(exe("rustdoc_tool_binary", target_compiler.host));

        // don't create a stage0-sysroot/bin directory.
        if target_compiler.stage > 0 {
            let bin_rustdoc = bin_rustdoc();
            builder.copy_link(&tool_rustdoc, &bin_rustdoc);
            bin_rustdoc
        } else {
            tool_rustdoc
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct Cargo {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for Cargo {
    type Output = PathBuf;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.path("src/tools/cargo").default_condition(builder.tool_enabled("cargo"))
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(Cargo {
            compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.build),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) -> PathBuf {
        builder.build.require_submodule("src/tools/cargo", None);

        builder.ensure(ToolBuild {
            compiler: self.compiler,
            target: self.target,
            tool: "cargo",
            mode: Mode::ToolRustc,
            path: "src/tools/cargo",
            source_type: SourceType::Submodule,
            extra_features: Vec::new(),
            allow_features: "",
            cargo_args: Vec::new(),
        })
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct LldWrapper {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for LldWrapper {
    type Output = PathBuf;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        run.never()
    }

    fn run(self, builder: &Builder<'_>) -> PathBuf {
        builder.ensure(ToolBuild {
            compiler: self.compiler,
            target: self.target,
            tool: "lld-wrapper",
            mode: Mode::ToolStd,
            path: "src/tools/lld-wrapper",
            source_type: SourceType::InTree,
            extra_features: Vec::new(),
            allow_features: "",
            cargo_args: Vec::new(),
        })
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
    type Output = PathBuf;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.path("src/tools/rust-analyzer").default_condition(builder.tool_enabled("rust-analyzer"))
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(RustAnalyzer {
            compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.build),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) -> PathBuf {
        builder.ensure(ToolBuild {
            compiler: self.compiler,
            target: self.target,
            tool: "rust-analyzer",
            mode: Mode::ToolRustc,
            path: "src/tools/rust-analyzer",
            extra_features: vec!["in-rust-tree".to_owned()],
            source_type: SourceType::InTree,
            allow_features: RustAnalyzer::ALLOW_FEATURES,
            cargo_args: Vec::new(),
        })
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RustAnalyzerProcMacroSrv {
    pub compiler: Compiler,
    pub target: TargetSelection,
}

impl Step for RustAnalyzerProcMacroSrv {
    type Output = Option<PathBuf>;
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
            compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.build),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) -> Option<PathBuf> {
        let path = builder.ensure(ToolBuild {
            compiler: self.compiler,
            target: self.target,
            tool: "rust-analyzer-proc-macro-srv",
            mode: Mode::ToolRustc,
            path: "src/tools/rust-analyzer/crates/proc-macro-srv-cli",
            extra_features: vec!["in-rust-tree".to_owned()],
            source_type: SourceType::InTree,
            allow_features: RustAnalyzer::ALLOW_FEATURES,
            cargo_args: Vec::new(),
        });

        // Copy `rust-analyzer-proc-macro-srv` to `<sysroot>/libexec/`
        // so that r-a can use it.
        let libexec_path = builder.sysroot(self.compiler).join("libexec");
        t!(fs::create_dir_all(&libexec_path));
        builder.copy_link(&path, &libexec_path.join("rust-analyzer-proc-macro-srv"));

        Some(path)
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct LlvmBitcodeLinker {
    pub compiler: Compiler,
    pub target: TargetSelection,
    pub extra_features: Vec<String>,
}

impl Step for LlvmBitcodeLinker {
    type Output = PathBuf;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
        let builder = run.builder;
        run.path("src/tools/llvm-bitcode-linker")
            .default_condition(builder.tool_enabled("llvm-bitcode-linker"))
    }

    fn make_run(run: RunConfig<'_>) {
        run.builder.ensure(LlvmBitcodeLinker {
            compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.build),
            extra_features: Vec::new(),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder<'_>) -> PathBuf {
        let bin_name = "llvm-bitcode-linker";

        // If enabled, use ci-rustc and skip building the in-tree compiler.
        if !builder.download_rustc() {
            builder.ensure(compile::Std::new(self.compiler, self.compiler.host));
            builder.ensure(compile::Rustc::new(self.compiler, self.target));
        }

        let cargo = prepare_tool_cargo(
            builder,
            self.compiler,
            Mode::ToolRustc,
            self.target,
            Kind::Build,
            "src/tools/llvm-bitcode-linker",
            SourceType::InTree,
            &self.extra_features,
        );

        let _guard = builder.msg_tool(
            Kind::Build,
            Mode::ToolRustc,
            bin_name,
            self.compiler.stage,
            &self.compiler.host,
            &self.target,
        );

        cargo.into_cmd().run(builder);

        let tool_out = builder
            .cargo_out(self.compiler, Mode::ToolRustc, self.target)
            .join(exe(bin_name, self.compiler.host));

        if self.compiler.stage > 0 {
            let bindir_self_contained = builder
                .sysroot(self.compiler)
                .join(format!("lib/rustlib/{}/bin/self-contained", self.target.triple));
            t!(fs::create_dir_all(&bindir_self_contained));
            let bin_destination = bindir_self_contained.join(exe(bin_name, self.compiler.host));
            builder.copy_link(&tool_out, &bin_destination);
            bin_destination
        } else {
            tool_out
        }
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct LibcxxVersionTool {
    pub target: TargetSelection,
}

#[allow(dead_code)]
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
    (($sel:ident, $builder:ident),
       $($name:ident,
       $path:expr,
       $tool_name:expr,
       stable = $stable:expr
       $(,tool_std = $tool_std:literal)?
       $(,allow_features = $allow_features:expr)?
       $(,add_bins_to_sysroot = $add_bins_to_sysroot:expr)?
       ;)+) => {
        $(
            #[derive(Debug, Clone, Hash, PartialEq, Eq)]
        pub struct $name {
            pub compiler: Compiler,
            pub target: TargetSelection,
            pub extra_features: Vec<String>,
        }

        impl Step for $name {
            type Output = PathBuf;
            const DEFAULT: bool = true; // Overwritten below
            const ONLY_HOSTS: bool = true;

            fn should_run(run: ShouldRun<'_>) -> ShouldRun<'_> {
                let builder = run.builder;
                run.path($path).default_condition(
                    builder.config.extended
                        && builder.config.tools.as_ref().map_or(
                            // By default, on nightly/dev enable all tools, else only
                            // build stable tools.
                            $stable || builder.build.unstable_features(),
                            // If `tools` is set, search list for this tool.
                            |tools| {
                                tools.iter().any(|tool| match tool.as_ref() {
                                    "clippy" => $tool_name == "clippy-driver",
                                    x => $tool_name == x,
                            })
                        }),
                )
            }

            fn make_run(run: RunConfig<'_>) {
                run.builder.ensure($name {
                    compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.build),
                    target: run.target,
                    extra_features: Vec::new(),
                });
            }

            #[allow(unused_mut)]
            fn run(mut $sel, $builder: &Builder<'_>) -> PathBuf {
                let tool = $builder.ensure(ToolBuild {
                    compiler: $sel.compiler,
                    target: $sel.target,
                    tool: $tool_name,
                    mode: if false $(|| $tool_std)? { Mode::ToolStd } else { Mode::ToolRustc },
                    path: $path,
                    extra_features: $sel.extra_features,
                    source_type: SourceType::InTree,
                    allow_features: concat!($($allow_features)*),
                    cargo_args: vec![]
                });

                if (false $(|| !$add_bins_to_sysroot.is_empty())?) && $sel.compiler.stage > 0 {
                    let bindir = $builder.sysroot($sel.compiler).join("bin");
                    t!(fs::create_dir_all(&bindir));

                    #[allow(unused_variables)]
                    let tools_out = $builder
                        .cargo_out($sel.compiler, Mode::ToolRustc, $sel.target);

                    $(for add_bin in $add_bins_to_sysroot {
                        let bin_source = tools_out.join(exe(add_bin, $sel.target));
                        let bin_destination = bindir.join(exe(add_bin, $sel.compiler.host));
                        $builder.copy_link(&bin_source, &bin_destination);
                    })?

                    let tool = bindir.join(exe($tool_name, $sel.compiler.host));
                    tool
                } else {
                    tool
                }
            }
        }
        )+
    }
}

tool_extended!((self, builder),
    Cargofmt, "src/tools/rustfmt", "cargo-fmt", stable=true;
    CargoClippy, "src/tools/clippy", "cargo-clippy", stable=true;
    Clippy, "src/tools/clippy", "clippy-driver", stable=true, add_bins_to_sysroot = ["clippy-driver", "cargo-clippy"];
    Miri, "src/tools/miri", "miri", stable=false, add_bins_to_sysroot = ["miri"];
    CargoMiri, "src/tools/miri/cargo-miri", "cargo-miri", stable=false, add_bins_to_sysroot = ["cargo-miri"];
    Rls, "src/tools/rls", "rls", stable=true;
    Rustfmt, "src/tools/rustfmt", "rustfmt", stable=true, add_bins_to_sysroot = ["rustfmt", "cargo-fmt"];
);

impl<'a> Builder<'a> {
    /// Gets a `BootstrapCommand` which is ready to run `tool` in `stage` built for
    /// `host`.
    pub fn tool_cmd(&self, tool: Tool) -> BootstrapCommand {
        let mut cmd = command(self.tool_exe(tool));
        let compiler = self.compiler(0, self.config.build);
        let host = &compiler.host;
        // Prepares the `cmd` provided to be able to run the `compiler` provided.
        //
        // Notably this munges the dynamic library lookup path to point to the
        // right location to run `compiler`.
        let mut lib_paths: Vec<PathBuf> = vec![
            self.build.rustc_snapshot_libdir(),
            self.cargo_out(compiler, Mode::ToolBootstrap, *host).join("deps"),
        ];

        // On MSVC a tool may invoke a C compiler (e.g., compiletest in run-make
        // mode) and that C compiler may need some extra PATH modification. Do
        // so here.
        if compiler.host.is_msvc() {
            let curpaths = env::var_os("PATH").unwrap_or_default();
            let curpaths = env::split_paths(&curpaths).collect::<Vec<_>>();
            for (k, v) in self.cc.borrow()[&compiler.host].env() {
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
