use std::fs;
use std::env;
use std::iter;
use std::path::PathBuf;
use std::process::{Command, exit};
use std::collections::HashSet;

use crate::Mode;
use crate::Compiler;
use crate::builder::{Step, RunConfig, ShouldRun, Builder};
use crate::util::{exe, add_lib_path};
use crate::compile;
use crate::native;
use crate::channel::GitInfo;
use crate::channel;
use crate::cache::Interned;
use crate::toolstate::ToolState;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum SourceType {
    InTree,
    Submodule,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct ToolBuild {
    compiler: Compiler,
    target: Interned<String>,
    tool: &'static str,
    path: &'static str,
    mode: Mode,
    is_optional_tool: bool,
    source_type: SourceType,
    extra_features: Vec<String>,
}

impl Step for ToolBuild {
    type Output = Option<PathBuf>;

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.never()
    }

    /// Build a tool in `src/tools`
    ///
    /// This will build the specified tool with the specified `host` compiler in
    /// `stage` into the normal cargo output directory.
    fn run(self, builder: &Builder) -> Option<PathBuf> {
        let compiler = self.compiler;
        let target = self.target;
        let tool = self.tool;
        let path = self.path;
        let is_optional_tool = self.is_optional_tool;

        match self.mode {
            Mode::ToolRustc => {
                builder.ensure(compile::Rustc { compiler, target })
            }
            Mode::ToolStd => {
                builder.ensure(compile::Std { compiler, target })
            }
            Mode::ToolBootstrap => {} // uses downloaded stage0 compiler libs
            _ => panic!("unexpected Mode for tool build")
        }

        let mut cargo = prepare_tool_cargo(
            builder,
            compiler,
            self.mode,
            target,
            "build",
            path,
            self.source_type,
            &self.extra_features,
        );

        let _folder = builder.fold_output(|| format!("stage{}-{}", compiler.stage, tool));
        builder.info(&format!("Building stage{} tool {} ({})", compiler.stage, tool, target));
        let mut duplicates = Vec::new();
        let is_expected = compile::stream_cargo(builder, &mut cargo, vec![], &mut |msg| {
            // Only care about big things like the RLS/Cargo for now
            match tool {
                | "rls"
                | "cargo"
                | "clippy-driver"
                => {}

                _ => return,
            }
            let (id, features, filenames) = match msg {
                compile::CargoMessage::CompilerArtifact {
                    package_id,
                    features,
                    filenames
                } => {
                    (package_id, features, filenames)
                }
                _ => return,
            };
            let features = features.iter().map(|s| s.to_string()).collect::<Vec<_>>();

            for path in filenames {
                let val = (tool, PathBuf::from(&*path), features.clone());
                // we're only interested in deduplicating rlibs for now
                if val.1.extension().and_then(|s| s.to_str()) != Some("rlib") {
                    continue
                }

                // Don't worry about libs that turn out to be host dependencies
                // or build scripts, we only care about target dependencies that
                // are in `deps`.
                if let Some(maybe_target) = val.1
                    .parent()                   // chop off file name
                    .and_then(|p| p.parent())   // chop off `deps`
                    .and_then(|p| p.parent())   // chop off `release`
                    .and_then(|p| p.file_name())
                    .and_then(|p| p.to_str())
                {
                    if maybe_target != &*target {
                        continue
                    }
                }

                let mut artifacts = builder.tool_artifacts.borrow_mut();
                let prev_artifacts = artifacts
                    .entry(target)
                    .or_default();
                if let Some(prev) = prev_artifacts.get(&*id) {
                    if prev.1 != val.1 {
                        duplicates.push((
                            id.to_string(),
                            val,
                            prev.clone(),
                        ));
                    }
                    return
                }
                prev_artifacts.insert(id.to_string(), val);
            }
        });

        if is_expected && !duplicates.is_empty() {
            println!("duplicate artfacts found when compiling a tool, this \
                      typically means that something was recompiled because \
                      a transitive dependency has different features activated \
                      than in a previous build:\n");
            println!("the following dependencies are duplicated although they \
                      have the same features enabled:");
            for (id, cur, prev) in duplicates.drain_filter(|(_, cur, prev)| cur.2 == prev.2) {
                println!("  {}", id);
                // same features
                println!("    `{}` ({:?})\n    `{}` ({:?})", cur.0, cur.1, prev.0, prev.1);
            }
            println!("the following dependencies have different features:");
            for (id, cur, prev) in duplicates {
                println!("  {}", id);
                let cur_features: HashSet<_> = cur.2.into_iter().collect();
                let prev_features: HashSet<_> = prev.2.into_iter().collect();
                println!("    `{}` additionally enabled features {:?} at {:?}",
                         cur.0, &cur_features - &prev_features, cur.1);
                println!("    `{}` additionally enabled features {:?} at {:?}",
                         prev.0, &prev_features - &cur_features, prev.1);
            }
            println!();
            println!("to fix this you will probably want to edit the local \
                      src/tools/rustc-workspace-hack/Cargo.toml crate, as \
                      that will update the dependency graph to ensure that \
                      these crates all share the same feature set");
            panic!("tools should not compile multiple copies of the same crate");
        }

        builder.save_toolstate(tool, if is_expected {
            ToolState::TestFail
        } else {
            ToolState::BuildFail
        });

        if !is_expected {
            if !is_optional_tool {
                exit(1);
            } else {
                None
            }
        } else {
            let cargo_out = builder.cargo_out(compiler, self.mode, target)
                .join(exe(tool, &compiler.host));
            let bin = builder.tools_dir(compiler).join(exe(tool, &compiler.host));
            builder.copy(&cargo_out, &bin);
            Some(bin)
        }
    }
}

pub fn prepare_tool_cargo(
    builder: &Builder,
    compiler: Compiler,
    mode: Mode,
    target: Interned<String>,
    command: &'static str,
    path: &'static str,
    source_type: SourceType,
    extra_features: &[String],
) -> Command {
    let mut cargo = builder.cargo(compiler, mode, target, command);
    let dir = builder.src.join(path);
    cargo.arg("--manifest-path").arg(dir.join("Cargo.toml"));

    // We don't want to build tools dynamically as they'll be running across
    // stages and such and it's just easier if they're not dynamically linked.
    cargo.env("RUSTC_NO_PREFER_DYNAMIC", "1");

    if source_type == SourceType::Submodule {
        cargo.env("RUSTC_EXTERNAL_TOOL", "1");
    }

    let mut features = extra_features.iter().cloned().collect::<Vec<_>>();
    if builder.build.config.cargo_native_static {
        if path.ends_with("cargo") ||
            path.ends_with("rls") ||
            path.ends_with("clippy") ||
            path.ends_with("rustfmt")
        {
            cargo.env("LIBZ_SYS_STATIC", "1");
            features.push("rustc-workspace-hack/all-static".to_string());
        }
    }

    // if tools are using lzma we want to force the build script to build its
    // own copy
    cargo.env("LZMA_API_STATIC", "1");

    cargo.env("CFG_RELEASE_CHANNEL", &builder.config.channel);
    cargo.env("CFG_VERSION", builder.rust_version());
    cargo.env("CFG_RELEASE_NUM", channel::CFG_RELEASE_NUM);

    let info = GitInfo::new(&builder.config, &dir);
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
        cargo.arg("--features").arg(&features.join(", "));
    }
    cargo
}

macro_rules! tool {
    ($(
        $name:ident, $path:expr, $tool_name:expr, $mode:expr
        $(,llvm_tools = $llvm:expr)*
        $(,is_external_tool = $external:expr)*
        ;
    )+) => {
        #[derive(Copy, PartialEq, Eq, Clone)]
        pub enum Tool {
            $(
                $name,
            )+
        }

        impl Tool {
            pub fn get_mode(&self) -> Mode {
                let mode = match self {
                    $(Tool::$name => $mode,)+
                };
                mode
            }

            /// Whether this tool requires LLVM to run
            pub fn uses_llvm_tools(&self) -> bool {
                match self {
                    $(Tool::$name => false $(|| $llvm)*,)+
                }
            }
        }

        impl<'a> Builder<'a> {
            pub fn tool_exe(&self, tool: Tool) -> PathBuf {
                let stage = self.tool_default_stage(tool);
                match tool {
                    $(Tool::$name =>
                        self.ensure($name {
                            compiler: self.compiler(stage, self.config.build),
                            target: self.config.build,
                        }),
                    )+
                }
            }

            pub fn tool_default_stage(&self, tool: Tool) -> u32 {
                // Compile the error-index in the same stage as rustdoc to avoid
                // recompiling rustdoc twice if we can. Otherwise compile
                // everything else in stage0 as there's no need to rebootstrap
                // everything.
                match tool {
                    Tool::ErrorIndex if self.top_stage >= 2 => self.top_stage,
                    _ => 0,
                }
            }
        }

        $(
            #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
        pub struct $name {
            pub compiler: Compiler,
            pub target: Interned<String>,
        }

        impl Step for $name {
            type Output = PathBuf;

            fn should_run(run: ShouldRun) -> ShouldRun {
                run.path($path)
            }

            fn make_run(run: RunConfig) {
                run.builder.ensure($name {
                    compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.build),
                    target: run.target,
                });
            }

            fn run(self, builder: &Builder) -> PathBuf {
                builder.ensure(ToolBuild {
                    compiler: self.compiler,
                    target: self.target,
                    tool: $tool_name,
                    mode: $mode,
                    path: $path,
                    is_optional_tool: false,
                    source_type: if false $(|| $external)* {
                        SourceType::Submodule
                    } else {
                        SourceType::InTree
                    },
                    extra_features: Vec::new(),
                }).expect("expected to build -- essential tool")
            }
        }
        )+
    }
}

tool!(
    Rustbook, "src/tools/rustbook", "rustbook", Mode::ToolBootstrap;
    ErrorIndex, "src/tools/error_index_generator", "error_index_generator", Mode::ToolRustc;
    UnstableBookGen, "src/tools/unstable-book-gen", "unstable-book-gen", Mode::ToolBootstrap;
    Tidy, "src/tools/tidy", "tidy", Mode::ToolBootstrap;
    Linkchecker, "src/tools/linkchecker", "linkchecker", Mode::ToolBootstrap;
    CargoTest, "src/tools/cargotest", "cargotest", Mode::ToolBootstrap;
    Compiletest, "src/tools/compiletest", "compiletest", Mode::ToolBootstrap, llvm_tools = true;
    BuildManifest, "src/tools/build-manifest", "build-manifest", Mode::ToolBootstrap;
    RemoteTestClient, "src/tools/remote-test-client", "remote-test-client", Mode::ToolBootstrap;
    RustInstaller, "src/tools/rust-installer", "fabricate", Mode::ToolBootstrap,
        is_external_tool = true;
    RustdocTheme, "src/tools/rustdoc-themes", "rustdoc-themes", Mode::ToolBootstrap;
);

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct RemoteTestServer {
    pub compiler: Compiler,
    pub target: Interned<String>,
}

impl Step for RemoteTestServer {
    type Output = PathBuf;

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("src/tools/remote-test-server")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(RemoteTestServer {
            compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.build),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder) -> PathBuf {
        builder.ensure(ToolBuild {
            compiler: self.compiler,
            target: self.target,
            tool: "remote-test-server",
            mode: Mode::ToolStd,
            path: "src/tools/remote-test-server",
            is_optional_tool: false,
            source_type: SourceType::InTree,
            extra_features: Vec::new(),
        }).expect("expected to build -- essential tool")
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Rustdoc {
    pub host: Interned<String>,
}

impl Step for Rustdoc {
    type Output = PathBuf;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("src/tools/rustdoc")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Rustdoc {
            host: run.host,
        });
    }

    fn run(self, builder: &Builder) -> PathBuf {
        let target_compiler = builder.compiler(builder.top_stage, self.host);
        let target = target_compiler.host;
        let build_compiler = if target_compiler.stage == 0 {
            builder.compiler(0, builder.config.build)
        } else if target_compiler.stage >= 2 {
            // Past stage 2, we consider the compiler to be ABI-compatible and hence capable of
            // building rustdoc itself.
            builder.compiler(target_compiler.stage, builder.config.build)
        } else {
            // Similar to `compile::Assemble`, build with the previous stage's compiler. Otherwise
            // we'd have stageN/bin/rustc and stageN/bin/rustdoc be effectively different stage
            // compilers, which isn't what we want.
            builder.compiler(target_compiler.stage - 1, builder.config.build)
        };

        builder.ensure(compile::Rustc { compiler: build_compiler, target });
        builder.ensure(compile::Rustc {
            compiler: build_compiler,
            target: builder.config.build,
        });

        let mut cargo = prepare_tool_cargo(
            builder,
            build_compiler,
            Mode::ToolRustc,
            target,
            "build",
            "src/tools/rustdoc",
            SourceType::InTree,
            &[],
        );

        // Most tools don't get debuginfo, but rustdoc should.
        cargo.env("RUSTC_DEBUGINFO", builder.config.rust_debuginfo.to_string())
             .env("RUSTC_DEBUGINFO_LINES", builder.config.rust_debuginfo_lines.to_string());

        let _folder = builder.fold_output(|| format!("stage{}-rustdoc", target_compiler.stage));
        builder.info(&format!("Building rustdoc for stage{} ({})",
            target_compiler.stage, target_compiler.host));
        builder.run(&mut cargo);

        // Cargo adds a number of paths to the dylib search path on windows, which results in
        // the wrong rustdoc being executed. To avoid the conflicting rustdocs, we name the "tool"
        // rustdoc a different name.
        let tool_rustdoc = builder.cargo_out(build_compiler, Mode::ToolRustc, target)
            .join(exe("rustdoc_tool_binary", &target_compiler.host));

        // don't create a stage0-sysroot/bin directory.
        if target_compiler.stage > 0 {
            let sysroot = builder.sysroot(target_compiler);
            let bindir = sysroot.join("bin");
            t!(fs::create_dir_all(&bindir));
            let bin_rustdoc = bindir.join(exe("rustdoc", &*target_compiler.host));
            let _ = fs::remove_file(&bin_rustdoc);
            builder.copy(&tool_rustdoc, &bin_rustdoc);
            bin_rustdoc
        } else {
            tool_rustdoc
        }
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Cargo {
    pub compiler: Compiler,
    pub target: Interned<String>,
}

impl Step for Cargo {
    type Output = PathBuf;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun) -> ShouldRun {
        let builder = run.builder;
        run.path("src/tools/cargo").default_condition(builder.config.extended)
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Cargo {
            compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.build),
            target: run.target,
        });
    }

    fn run(self, builder: &Builder) -> PathBuf {
        // Cargo depends on procedural macros, which requires a full host
        // compiler to be available, so we need to depend on that.
        builder.ensure(compile::Rustc {
            compiler: self.compiler,
            target: builder.config.build,
        });
        builder.ensure(ToolBuild {
            compiler: self.compiler,
            target: self.target,
            tool: "cargo",
            mode: Mode::ToolRustc,
            path: "src/tools/cargo",
            is_optional_tool: false,
            source_type: SourceType::Submodule,
            extra_features: Vec::new(),
        }).expect("expected to build -- essential tool")
    }
}

macro_rules! tool_extended {
    (($sel:ident, $builder:ident),
       $($name:ident,
       $toolstate:ident,
       $path:expr,
       $tool_name:expr,
       $extra_deps:block;)+) => {
        $(
            #[derive(Debug, Clone, Hash, PartialEq, Eq)]
        pub struct $name {
            pub compiler: Compiler,
            pub target: Interned<String>,
            pub extra_features: Vec<String>,
        }

        impl Step for $name {
            type Output = Option<PathBuf>;
            const DEFAULT: bool = true;
            const ONLY_HOSTS: bool = true;

            fn should_run(run: ShouldRun) -> ShouldRun {
                let builder = run.builder;
                run.path($path).default_condition(builder.config.extended)
            }

            fn make_run(run: RunConfig) {
                run.builder.ensure($name {
                    compiler: run.builder.compiler(run.builder.top_stage, run.builder.config.build),
                    target: run.target,
                    extra_features: Vec::new(),
                });
            }

            #[allow(unused_mut)]
            fn run(mut $sel, $builder: &Builder) -> Option<PathBuf> {
                $extra_deps
                $builder.ensure(ToolBuild {
                    compiler: $sel.compiler,
                    target: $sel.target,
                    tool: $tool_name,
                    mode: Mode::ToolRustc,
                    path: $path,
                    extra_features: $sel.extra_features,
                    is_optional_tool: true,
                    source_type: SourceType::Submodule,
                })
            }
        }
        )+
    }
}

tool_extended!((self, builder),
    Cargofmt, rustfmt, "src/tools/rustfmt", "cargo-fmt", {};
    CargoClippy, clippy, "src/tools/clippy", "cargo-clippy", {
        // Clippy depends on procedural macros (serde), which requires a full host
        // compiler to be available, so we need to depend on that.
        builder.ensure(compile::Rustc {
            compiler: self.compiler,
            target: builder.config.build,
        });
    };
    Clippy, clippy, "src/tools/clippy", "clippy-driver", {
        // Clippy depends on procedural macros (serde), which requires a full host
        // compiler to be available, so we need to depend on that.
        builder.ensure(compile::Rustc {
            compiler: self.compiler,
            target: builder.config.build,
        });
    };
    Miri, miri, "src/tools/miri", "miri", {};
    Rls, rls, "src/tools/rls", "rls", {
        let clippy = builder.ensure(Clippy {
            compiler: self.compiler,
            target: self.target,
            extra_features: Vec::new(),
        });
        if clippy.is_some() {
            self.extra_features.push("clippy".to_owned());
        }
        // RLS depends on procedural macros, which requires a full host
        // compiler to be available, so we need to depend on that.
        builder.ensure(compile::Rustc {
            compiler: self.compiler,
            target: builder.config.build,
        });
    };
    Rustfmt, rustfmt, "src/tools/rustfmt", "rustfmt", {};
);

impl<'a> Builder<'a> {
    /// Get a `Command` which is ready to run `tool` in `stage` built for
    /// `host`.
    pub fn tool_cmd(&self, tool: Tool) -> Command {
        let mut cmd = Command::new(self.tool_exe(tool));
        let compiler = self.compiler(self.tool_default_stage(tool), self.config.build);
        self.prepare_tool_cmd(compiler, tool, &mut cmd);
        cmd
    }

    /// Prepares the `cmd` provided to be able to run the `compiler` provided.
    ///
    /// Notably this munges the dynamic library lookup path to point to the
    /// right location to run `compiler`.
    fn prepare_tool_cmd(&self, compiler: Compiler, tool: Tool, cmd: &mut Command) {
        let host = &compiler.host;
        let mut lib_paths: Vec<PathBuf> = vec![
            if compiler.stage == 0 && tool != Tool::ErrorIndex {
                self.build.rustc_snapshot_libdir()
            } else {
                PathBuf::from(&self.sysroot_libdir(compiler, compiler.host))
            },
            self.cargo_out(compiler, tool.get_mode(), *host).join("deps"),
        ];

        // On MSVC a tool may invoke a C compiler (e.g., compiletest in run-make
        // mode) and that C compiler may need some extra PATH modification. Do
        // so here.
        if compiler.host.contains("msvc") {
            let curpaths = env::var_os("PATH").unwrap_or_default();
            let curpaths = env::split_paths(&curpaths).collect::<Vec<_>>();
            for &(ref k, ref v) in self.cc[&compiler.host].env() {
                if k != "PATH" {
                    continue
                }
                for path in env::split_paths(v) {
                    if !curpaths.contains(&path) {
                        lib_paths.push(path);
                    }
                }
            }
        }

        // Add the llvm/bin directory to PATH since it contains lots of
        // useful, platform-independent tools
        if tool.uses_llvm_tools() {
            if let Some(llvm_bin_path) = self.llvm_bin_path() {
                if host.contains("windows") {
                    // On Windows, PATH and the dynamic library path are the same,
                    // so we just add the LLVM bin path to lib_path
                    lib_paths.push(llvm_bin_path);
                } else {
                    let old_path = env::var_os("PATH").unwrap_or_default();
                    let new_path = env::join_paths(iter::once(llvm_bin_path)
                            .chain(env::split_paths(&old_path)))
                        .expect("Could not add LLVM bin path to PATH");
                    cmd.env("PATH", new_path);
                }
            }
        }

        add_lib_path(lib_paths, cmd);
    }

    fn llvm_bin_path(&self) -> Option<PathBuf> {
        if self.config.llvm_enabled && !self.config.dry_run {
            let llvm_config = self.ensure(native::Llvm {
                target: self.config.build,
                emscripten: false,
            });

            // Add the llvm/bin directory to PATH since it contains lots of
            // useful, platform-independent tools
            let llvm_bin_path = llvm_config.parent()
                .expect("Expected llvm-config to be contained in directory");
            assert!(llvm_bin_path.is_dir());
            Some(llvm_bin_path.to_path_buf())
        } else {
            None
        }
    }
}
