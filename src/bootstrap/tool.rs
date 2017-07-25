// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::env;
use std::path::PathBuf;
use std::process::Command;

use Mode;
use Compiler;
use builder::{Step, RunConfig, ShouldRun, Builder};
use util::{exe, add_lib_path};
use compile::{self, libtest_stamp, libstd_stamp, librustc_stamp};
use native;
use channel::GitInfo;
use cache::Interned;

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct CleanTools {
    pub stage: u32,
    pub target: Interned<String>,
    pub mode: Mode,
}

impl Step for CleanTools {
    type Output = ();

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.never()
    }

    /// Build a tool in `src/tools`
    ///
    /// This will build the specified tool with the specified `host` compiler in
    /// `stage` into the normal cargo output directory.
    fn run(self, builder: &Builder) {
        let build = builder.build;
        let stage = self.stage;
        let target = self.target;
        let mode = self.mode;

        let compiler = builder.compiler(stage, build.build);

        let stamp = match mode {
            Mode::Libstd => libstd_stamp(build, compiler, target),
            Mode::Libtest => libtest_stamp(build, compiler, target),
            Mode::Librustc => librustc_stamp(build, compiler, target),
            _ => panic!(),
        };
        let out_dir = build.cargo_out(compiler, Mode::Tool, target);
        build.clear_if_dirty(&out_dir, &stamp);
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct ToolBuild {
    pub stage: u32,
    pub target: Interned<String>,
    pub tool: &'static str,
    pub mode: Mode,
}

impl Step for ToolBuild {
    type Output = PathBuf;

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.never()
    }

    /// Build a tool in `src/tools`
    ///
    /// This will build the specified tool with the specified `host` compiler in
    /// `stage` into the normal cargo output directory.
    fn run(self, builder: &Builder) -> PathBuf {
        let build = builder.build;
        let stage = self.stage;
        let target = self.target;
        let tool = self.tool;

        let compiler = builder.compiler(stage, build.build);
        builder.ensure(CleanTools { stage, target, mode: self.mode });
        match self.mode {
            Mode::Libstd => builder.ensure(compile::Std { compiler, target }),
            Mode::Libtest => builder.ensure(compile::Test { compiler, target }),
            Mode::Librustc => builder.ensure(compile::Rustc { compiler, target }),
            Mode::Tool => panic!("unexpected Mode::Tool for tool build")
        }

        let _folder = build.fold_output(|| format!("stage{}-{}", stage, tool));
        println!("Building stage{} tool {} ({})", stage, tool, target);

        let mut cargo = builder.cargo(compiler, Mode::Tool, target, "build");
        let dir = build.src.join("src/tools").join(tool);
        cargo.arg("--manifest-path").arg(dir.join("Cargo.toml"));

        // We don't want to build tools dynamically as they'll be running across
        // stages and such and it's just easier if they're not dynamically linked.
        cargo.env("RUSTC_NO_PREFER_DYNAMIC", "1");

        if let Some(dir) = build.openssl_install_dir(target) {
            cargo.env("OPENSSL_STATIC", "1");
            cargo.env("OPENSSL_DIR", dir);
            cargo.env("LIBZ_SYS_STATIC", "1");
        }

        cargo.env("CFG_RELEASE_CHANNEL", &build.config.channel);

        let info = GitInfo::new(&dir);
        if let Some(sha) = info.sha() {
            cargo.env("CFG_COMMIT_HASH", sha);
        }
        if let Some(sha_short) = info.sha_short() {
            cargo.env("CFG_SHORT_COMMIT_HASH", sha_short);
        }
        if let Some(date) = info.commit_date() {
            cargo.env("CFG_COMMIT_DATE", date);
        }

        build.run(&mut cargo);
        build.cargo_out(compiler, Mode::Tool, target).join(exe(tool, &compiler.host))
    }
}

macro_rules! tool {
    ($($name:ident, $path:expr, $tool_name:expr, $mode:expr;)+) => {
        #[derive(Copy, Clone)]
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
                            stage: 0,
                            target: self.build.build,
                        }),
                    )+
                }
            }
        }

        $(
            #[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
        pub struct $name {
            pub stage: u32,
            pub target: Interned<String>,
        }

        impl Step for $name {
            type Output = PathBuf;

            fn should_run(run: ShouldRun) -> ShouldRun {
                run.path($path)
            }

            fn make_run(run: RunConfig) {
                run.builder.ensure($name {
                    stage: run.builder.top_stage,
                    target: run.target,
                });
            }

            fn run(self, builder: &Builder) -> PathBuf {
                builder.ensure(ToolBuild {
                    stage: self.stage,
                    target: self.target,
                    tool: $tool_name,
                    mode: $mode,
                })
            }
        }
        )+
    }
}

tool!(
    Rustbook, "src/tools/rustbook", "rustbook", Mode::Librustc;
    ErrorIndex, "src/tools/error_index_generator", "error_index_generator", Mode::Librustc;
    UnstableBookGen, "src/tools/unstable-book-gen", "unstable-book-gen", Mode::Libstd;
    Tidy, "src/tools/tidy", "tidy", Mode::Libstd;
    Linkchecker, "src/tools/linkchecker", "linkchecker", Mode::Libstd;
    CargoTest, "src/tools/cargotest", "cargotest", Mode::Libstd;
    Compiletest, "src/tools/compiletest", "compiletest", Mode::Libtest;
    BuildManifest, "src/tools/build-manifest", "build-manifest", Mode::Librustc;
    RemoteTestClient, "src/tools/remote-test-client", "remote-test-client", Mode::Libstd;
    RustInstaller, "src/tools/rust-installer", "rust-installer", Mode::Libstd;
);

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct RemoteTestServer {
    pub stage: u32,
    pub target: Interned<String>,
}

impl Step for RemoteTestServer {
    type Output = PathBuf;

    fn should_run(run: ShouldRun) -> ShouldRun {
        run.path("src/tools/remote-test-server")
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(RemoteTestServer {
            stage: run.builder.top_stage,
            target: run.target,
        });
    }

    fn run(self, builder: &Builder) -> PathBuf {
        builder.ensure(ToolBuild {
            stage: self.stage,
            target: self.target,
            tool: "remote-test-server",
            mode: Mode::Libstd,
        })
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Cargo {
    pub stage: u32,
    pub target: Interned<String>,
}

impl Step for Cargo {
    type Output = PathBuf;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun) -> ShouldRun {
        let builder = run.builder;
        run.path("src/tools/cargo").default_condition(builder.build.config.extended)
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Cargo {
            stage: run.builder.top_stage,
            target: run.target,
        });
    }

    fn run(self, builder: &Builder) -> PathBuf {
        builder.ensure(native::Openssl {
            target: self.target,
        });
        // Cargo depends on procedural macros, which requires a full host
        // compiler to be available, so we need to depend on that.
        builder.ensure(compile::Rustc {
            compiler: builder.compiler(self.stage, builder.build.build),
            target: builder.build.build,
        });
        builder.ensure(ToolBuild {
            stage: self.stage,
            target: self.target,
            tool: "cargo",
            mode: Mode::Librustc,
        })
    }
}

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct Rls {
    pub stage: u32,
    pub target: Interned<String>,
}

impl Step for Rls {
    type Output = PathBuf;
    const DEFAULT: bool = true;
    const ONLY_HOSTS: bool = true;

    fn should_run(run: ShouldRun) -> ShouldRun {
        let builder = run.builder;
        run.path("src/tools/rls").default_condition(builder.build.config.extended)
    }

    fn make_run(run: RunConfig) {
        run.builder.ensure(Rls {
            stage: run.builder.top_stage,
            target: run.target,
        });
    }

    fn run(self, builder: &Builder) -> PathBuf {
        builder.ensure(native::Openssl {
            target: self.target,
        });
        // RLS depends on procedural macros, which requires a full host
        // compiler to be available, so we need to depend on that.
        builder.ensure(compile::Rustc {
            compiler: builder.compiler(self.stage, builder.build.build),
            target: builder.build.build,
        });
        builder.ensure(ToolBuild {
            stage: self.stage,
            target: self.target,
            tool: "rls",
            mode: Mode::Librustc,
        })
    }
}

impl<'a> Builder<'a> {
    /// Get a `Command` which is ready to run `tool` in `stage` built for
    /// `host`.
    pub fn tool_cmd(&self, tool: Tool) -> Command {
        let mut cmd = Command::new(self.tool_exe(tool));
        let compiler = self.compiler(0, self.build.build);
        self.prepare_tool_cmd(compiler, &mut cmd);
        cmd
    }

    /// Prepares the `cmd` provided to be able to run the `compiler` provided.
    ///
    /// Notably this munges the dynamic library lookup path to point to the
    /// right location to run `compiler`.
    fn prepare_tool_cmd(&self, compiler: Compiler, cmd: &mut Command) {
        let host = &compiler.host;
        let mut paths: Vec<PathBuf> = vec![
            PathBuf::from(&self.sysroot_libdir(compiler, compiler.host)),
            self.cargo_out(compiler, Mode::Tool, *host).join("deps"),
        ];

        // On MSVC a tool may invoke a C compiler (e.g. compiletest in run-make
        // mode) and that C compiler may need some extra PATH modification. Do
        // so here.
        if compiler.host.contains("msvc") {
            let curpaths = env::var_os("PATH").unwrap_or_default();
            let curpaths = env::split_paths(&curpaths).collect::<Vec<_>>();
            for &(ref k, ref v) in self.cc[&compiler.host].0.env() {
                if k != "PATH" {
                    continue
                }
                for path in env::split_paths(v) {
                    if !curpaths.contains(&path) {
                        paths.push(path);
                    }
                }
            }
        }
        add_lib_path(paths, cmd);
    }
}
