// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Major workhorse of rustbuild, definition and dependencies between stages of
//! the copmile.
//!
//! The primary purpose of this module is to define the various `Step`s of
//! execution of the build. Each `Step` has a corresponding `Source` indicating
//! what it's actually doing along with a number of dependencies which must be
//! executed first.
//!
//! This module will take the CLI as input and calculate the steps required for
//! the build requested, ensuring that all intermediate pieces are in place.
//! Essentially this module is a `make`-replacement, but not as good.

use std::collections::HashSet;

use {Build, Compiler};

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct Step<'a> {
    pub src: Source<'a>,
    pub target: &'a str,
}

/// Macro used to iterate over all targets that are recognized by the build
/// system.
///
/// Whenever a new step is added it will involve adding an entry here, updating
/// the dependencies section below, and then adding an implementation of the
/// step in `build/mod.rs`.
///
/// This macro takes another macro as an argument and then calls that macro with
/// all steps that the build system knows about.
macro_rules! targets {
    ($m:ident) => {
        $m! {
            // Step representing building the stageN compiler. This is just the
            // compiler executable itself, not any of the support libraries
            (rustc, Rustc { stage: u32 }),

            // Steps for the two main cargo builds. These are parameterized over
            // the compiler which is producing the artifact.
            (libstd, Libstd { compiler: Compiler<'a> }),
            (libtest, Libtest { compiler: Compiler<'a> }),
            (librustc, Librustc { compiler: Compiler<'a> }),

            // Links the target produced by the compiler provided into the
            // host's directory also provided.
            (libstd_link, LibstdLink {
                compiler: Compiler<'a>,
                host: &'a str
            }),
            (libtest_link, LibtestLink {
                compiler: Compiler<'a>,
                host: &'a str
            }),
            (librustc_link, LibrustcLink {
                compiler: Compiler<'a>,
                host: &'a str
            }),

            // Various tools that we can build as part of the build.
            (tool_linkchecker, ToolLinkchecker { stage: u32 }),
            (tool_rustbook, ToolRustbook { stage: u32 }),
            (tool_error_index, ToolErrorIndex { stage: u32 }),
            (tool_cargotest, ToolCargoTest { stage: u32 }),
            (tool_tidy, ToolTidy { stage: u32 }),
            (tool_compiletest, ToolCompiletest { stage: u32 }),

            // Steps for long-running native builds. Ideally these wouldn't
            // actually exist and would be part of build scripts, but for now
            // these are here.
            //
            // There aren't really any parameters to this, but empty structs
            // with braces are unstable so we just pick something that works.
            (llvm, Llvm { _dummy: () }),
            (compiler_rt, CompilerRt { _dummy: () }),
            (test_helpers, TestHelpers { _dummy: () }),
            (debugger_scripts, DebuggerScripts { stage: u32 }),

            // Steps for various pieces of documentation that we can generate,
            // the 'doc' step is just a pseudo target to depend on a bunch of
            // others.
            (doc, Doc { stage: u32 }),
            (doc_book, DocBook { stage: u32 }),
            (doc_nomicon, DocNomicon { stage: u32 }),
            (doc_standalone, DocStandalone { stage: u32 }),
            (doc_std, DocStd { stage: u32 }),
            (doc_test, DocTest { stage: u32 }),
            (doc_rustc, DocRustc { stage: u32 }),
            (doc_error_index, DocErrorIndex { stage: u32 }),

            // Steps for running tests. The 'check' target is just a pseudo
            // target to depend on a bunch of others.
            (check, Check { stage: u32, compiler: Compiler<'a> }),
            (check_target, CheckTarget { stage: u32, compiler: Compiler<'a> }),
            (check_linkcheck, CheckLinkcheck { stage: u32 }),
            (check_cargotest, CheckCargoTest { stage: u32 }),
            (check_tidy, CheckTidy { stage: u32 }),
            (check_rpass, CheckRPass { compiler: Compiler<'a> }),
            (check_rpass_full, CheckRPassFull { compiler: Compiler<'a> }),
            (check_rpass_valgrind, CheckRPassValgrind { compiler: Compiler<'a> }),
            (check_rfail, CheckRFail { compiler: Compiler<'a> }),
            (check_rfail_full, CheckRFailFull { compiler: Compiler<'a> }),
            (check_cfail, CheckCFail { compiler: Compiler<'a> }),
            (check_cfail_full, CheckCFailFull { compiler: Compiler<'a> }),
            (check_pfail, CheckPFail { compiler: Compiler<'a> }),
            (check_pretty, CheckPretty { compiler: Compiler<'a> }),
            (check_pretty_rpass, CheckPrettyRPass { compiler: Compiler<'a> }),
            (check_pretty_rpass_full, CheckPrettyRPassFull { compiler: Compiler<'a> }),
            (check_pretty_rfail, CheckPrettyRFail { compiler: Compiler<'a> }),
            (check_pretty_rfail_full, CheckPrettyRFailFull { compiler: Compiler<'a> }),
            (check_pretty_rpass_valgrind, CheckPrettyRPassValgrind { compiler: Compiler<'a> }),
            (check_codegen, CheckCodegen { compiler: Compiler<'a> }),
            (check_codegen_units, CheckCodegenUnits { compiler: Compiler<'a> }),
            (check_incremental, CheckIncremental { compiler: Compiler<'a> }),
            (check_ui, CheckUi { compiler: Compiler<'a> }),
            (check_mir_opt, CheckMirOpt { compiler: Compiler<'a> }),
            (check_debuginfo, CheckDebuginfo { compiler: Compiler<'a> }),
            (check_rustdoc, CheckRustdoc { compiler: Compiler<'a> }),
            (check_docs, CheckDocs { compiler: Compiler<'a> }),
            (check_error_index, CheckErrorIndex { compiler: Compiler<'a> }),
            (check_rmake, CheckRMake { compiler: Compiler<'a> }),
            (check_crate_std, CheckCrateStd { compiler: Compiler<'a> }),
            (check_crate_test, CheckCrateTest { compiler: Compiler<'a> }),
            (check_crate_rustc, CheckCrateRustc { compiler: Compiler<'a> }),

            // Distribution targets, creating tarballs
            (dist, Dist { stage: u32 }),
            (dist_docs, DistDocs { stage: u32 }),
            (dist_mingw, DistMingw { _dummy: () }),
            (dist_rustc, DistRustc { stage: u32 }),
            (dist_std, DistStd { compiler: Compiler<'a> }),
            (dist_src, DistSrc { _dummy: () }),

            // Misc targets
            (android_copy_libs, AndroidCopyLibs { compiler: Compiler<'a> }),
        }
    }
}

// Define the `Source` enum by iterating over all the steps and peeling out just
// the types that we want to define.

macro_rules! item { ($a:item) => ($a) }

macro_rules! define_source {
    ($(($short:ident, $name:ident { $($args:tt)* }),)*) => {
        item! {
            #[derive(Hash, Eq, PartialEq, Clone, Debug)]
            pub enum Source<'a> {
                $($name { $($args)* }),*
            }
        }
    }
}

targets!(define_source);

/// Calculate a list of all steps described by `build`.
///
/// This will inspect the flags passed in on the command line and use that to
/// build up a list of steps to execute. These steps will then be transformed
/// into a topologically sorted list which when executed left-to-right will
/// correctly sequence the entire build.
pub fn all(build: &Build) -> Vec<Step> {
    let mut ret = Vec::new();
    let mut all = HashSet::new();
    for target in top_level(build) {
        fill(build, &target, &mut ret, &mut all);
    }
    return ret;

    fn fill<'a>(build: &'a Build,
                target: &Step<'a>,
                ret: &mut Vec<Step<'a>>,
                set: &mut HashSet<Step<'a>>) {
        if set.insert(target.clone()) {
            for dep in target.deps(build) {
                fill(build, &dep, ret, set);
            }
            ret.push(target.clone());
        }
    }
}

/// Determines what top-level targets are requested as part of this build,
/// returning them as a list.
fn top_level(build: &Build) -> Vec<Step> {
    let mut targets = Vec::new();
    let stage = build.flags.stage.unwrap_or(2);

    let host = Step {
        src: Source::Llvm { _dummy: () },
        target: build.flags.host.iter().next()
                     .unwrap_or(&build.config.build),
    };
    let target = Step {
        src: Source::Llvm { _dummy: () },
        target: build.flags.target.iter().next().map(|x| &x[..])
                     .unwrap_or(host.target)
    };

    // First, try to find steps on the command line.
    add_steps(build, stage, &host, &target, &mut targets);

    // If none are specified, then build everything.
    if targets.len() == 0 {
        let t = Step {
            src: Source::Llvm { _dummy: () },
            target: &build.config.build,
        };
        if build.config.docs {
          targets.push(t.doc(stage));
        }
        for host in build.config.host.iter() {
            if !build.flags.host.contains(host) {
                continue
            }
            let host = t.target(host);
            if host.target == build.config.build {
                targets.push(host.librustc(host.compiler(stage)));
            } else {
                targets.push(host.librustc_link(t.compiler(stage), host.target));
            }
            for target in build.config.target.iter() {
                if !build.flags.target.contains(target) {
                    continue
                }

                if host.target == build.config.build {
                    targets.push(host.target(target)
                                     .libtest(host.compiler(stage)));
                } else {
                    targets.push(host.target(target)
                                     .libtest_link(t.compiler(stage), host.target));
                }
            }
        }
    }

    return targets

}

fn add_steps<'a>(build: &'a Build,
                 stage: u32,
                 host: &Step<'a>,
                 target: &Step<'a>,
                 targets: &mut Vec<Step<'a>>) {
    struct Context<'a> {
        stage: u32,
        compiler: Compiler<'a>,
        _dummy: (),
        host: &'a str,
    }
    for step in build.flags.step.iter() {

        // The macro below insists on hygienic access to all local variables, so
        // we shove them all in a struct and subvert hygiene by accessing struct
        // fields instead,
        let cx = Context {
            stage: stage,
            compiler: host.target(&build.config.build).compiler(stage),
            _dummy: (),
            host: host.target,
        };
        macro_rules! add_step {
            ($(($short:ident, $name:ident { $($arg:ident: $t:ty),* }),)*) => ({$(
                let name = stringify!($short).replace("_", "-");
                if &step[..] == &name[..] {
                    targets.push(target.$short($(cx.$arg),*));
                    continue
                }
                drop(name);
            )*})
        }

        targets!(add_step);

        panic!("unknown step: {}", step);
    }
}

macro_rules! constructors {
    ($(($short:ident, $name:ident { $($arg:ident: $t:ty),* }),)*) => {$(
        fn $short(&self, $($arg: $t),*) -> Step<'a> {
            Step {
                src: Source::$name { $($arg: $arg),* },
                target: self.target,
            }
        }
    )*}
}

impl<'a> Step<'a> {
    fn compiler(&self, stage: u32) -> Compiler<'a> {
        Compiler::new(stage, self.target)
    }

    fn target(&self, target: &'a str) -> Step<'a> {
        Step { target: target, src: self.src.clone() }
    }

    // Define ergonomic constructors for each step defined above so they can be
    // easily constructed.
    targets!(constructors);

    /// Mapping of all dependencies for rustbuild.
    ///
    /// This function receives a step, the build that we're building for, and
    /// then returns a list of all the dependencies of that step.
    pub fn deps(&self, build: &'a Build) -> Vec<Step<'a>> {
        match self.src {
            Source::Rustc { stage: 0 } => {
                Vec::new()
            }
            Source::Rustc { stage } => {
                let compiler = Compiler::new(stage - 1, &build.config.build);
                vec![self.librustc(compiler)]
            }
            Source::Librustc { compiler } => {
                vec![self.libtest(compiler), self.llvm(())]
            }
            Source::Libtest { compiler } => {
                vec![self.libstd(compiler)]
            }
            Source::Libstd { compiler } => {
                vec![self.compiler_rt(()),
                     self.rustc(compiler.stage).target(compiler.host)]
            }
            Source::LibrustcLink { compiler, host } => {
                vec![self.librustc(compiler),
                     self.libtest_link(compiler, host)]
            }
            Source::LibtestLink { compiler, host } => {
                vec![self.libtest(compiler), self.libstd_link(compiler, host)]
            }
            Source::LibstdLink { compiler, host } => {
                vec![self.libstd(compiler),
                     self.target(host).rustc(compiler.stage)]
            }
            Source::CompilerRt { _dummy } => Vec::new(),
            Source::Llvm { _dummy } => Vec::new(),
            Source::TestHelpers { _dummy } => Vec::new(),
            Source::DebuggerScripts { stage: _ } => Vec::new(),

            // Note that all doc targets depend on artifacts from the build
            // architecture, not the target (which is where we're generating
            // docs into).
            Source::DocStd { stage } => {
                let compiler = self.target(&build.config.build).compiler(stage);
                vec![self.libstd(compiler)]
            }
            Source::DocTest { stage } => {
                let compiler = self.target(&build.config.build).compiler(stage);
                vec![self.libtest(compiler)]
            }
            Source::DocBook { stage } |
            Source::DocNomicon { stage } => {
                vec![self.target(&build.config.build).tool_rustbook(stage)]
            }
            Source::DocErrorIndex { stage } => {
                vec![self.target(&build.config.build).tool_error_index(stage)]
            }
            Source::DocStandalone { stage } => {
                vec![self.target(&build.config.build).rustc(stage)]
            }
            Source::DocRustc { stage } => {
                vec![self.doc_test(stage)]
            }
            Source::Doc { stage } => {
                let mut deps = vec![
                    self.doc_book(stage), self.doc_nomicon(stage),
                    self.doc_standalone(stage), self.doc_std(stage),
                    self.doc_error_index(stage),
                ];

                if build.config.compiler_docs {
                    deps.push(self.doc_rustc(stage));
                }

                deps
            }
            Source::Check { stage, compiler } => {
                // Check is just a pseudo step which means check all targets,
                // so just depend on checking all targets.
                build.config.target.iter().map(|t| {
                    self.target(t).check_target(stage, compiler)
                }).collect()
            }
            Source::CheckTarget { stage, compiler } => {
                // CheckTarget here means run all possible test suites for this
                // target. Most of the time, however, we can't actually run
                // anything if we're not the build triple as we could be cross
                // compiling.
                //
                // As a result, the base set of targets here is quite stripped
                // down from the standard set of targets. These suites have
                // their own internal logic to run in cross-compiled situations
                // if they'll run at all. For example compiletest knows that
                // when testing Android targets we ship artifacts to the
                // emulator.
                //
                // When in doubt the rule of thumb for adding to this list is
                // "should this test suite run on the android bot?"
                let mut base = vec![
                    self.check_rpass(compiler),
                    self.check_rfail(compiler),
                    self.check_crate_std(compiler),
                    self.check_crate_test(compiler),
                    self.check_debuginfo(compiler),
                    self.dist(stage),
                ];

                // If we're testing the build triple, then we know we can
                // actually run binaries and such, so we run all possible tests
                // that we know about.
                if self.target == build.config.build {
                    base.extend(vec![
                        // docs-related
                        self.check_docs(compiler),
                        self.check_error_index(compiler),
                        self.check_rustdoc(compiler),

                        // UI-related
                        self.check_cfail(compiler),
                        self.check_pfail(compiler),
                        self.check_ui(compiler),

                        // codegen-related
                        self.check_incremental(compiler),
                        self.check_codegen(compiler),
                        self.check_codegen_units(compiler),

                        // misc compiletest-test suites
                        self.check_rpass_full(compiler),
                        self.check_rfail_full(compiler),
                        self.check_cfail_full(compiler),
                        self.check_pretty_rpass_full(compiler),
                        self.check_pretty_rfail_full(compiler),
                        self.check_rpass_valgrind(compiler),
                        self.check_rmake(compiler),
                        self.check_mir_opt(compiler),

                        // crates
                        self.check_crate_rustc(compiler),

                        // pretty
                        self.check_pretty(compiler),
                        self.check_pretty_rpass(compiler),
                        self.check_pretty_rfail(compiler),
                        self.check_pretty_rpass_valgrind(compiler),

                        // misc
                        self.check_linkcheck(stage),
                        self.check_tidy(stage),
                    ]);
                }
                return base
            }
            Source::CheckLinkcheck { stage } => {
                vec![self.tool_linkchecker(stage), self.doc(stage)]
            }
            Source::CheckCargoTest { stage } => {
                vec![self.tool_cargotest(stage),
                     self.librustc(self.compiler(stage))]
            }
            Source::CheckTidy { stage } => {
                vec![self.tool_tidy(stage)]
            }
            Source::CheckMirOpt { compiler} |
            Source::CheckPrettyRPass { compiler } |
            Source::CheckPrettyRFail { compiler } |
            Source::CheckRFail { compiler } |
            Source::CheckPFail { compiler } |
            Source::CheckCodegen { compiler } |
            Source::CheckCodegenUnits { compiler } |
            Source::CheckIncremental { compiler } |
            Source::CheckUi { compiler } |
            Source::CheckRustdoc { compiler } |
            Source::CheckPretty { compiler } |
            Source::CheckCFail { compiler } |
            Source::CheckRPassValgrind { compiler } |
            Source::CheckRPass { compiler } => {
                let mut base = vec![
                    self.libtest(compiler),
                    self.target(compiler.host).tool_compiletest(compiler.stage),
                    self.test_helpers(()),
                ];
                if self.target.contains("android") {
                    base.push(self.android_copy_libs(compiler));
                }
                base
            }
            Source::CheckDebuginfo { compiler } => {
                vec![
                    self.libtest(compiler),
                    self.target(compiler.host).tool_compiletest(compiler.stage),
                    self.test_helpers(()),
                    self.debugger_scripts(compiler.stage),
                ]
            }
            Source::CheckRPassFull { compiler } |
            Source::CheckRFailFull { compiler } |
            Source::CheckCFailFull { compiler } |
            Source::CheckPrettyRPassFull { compiler } |
            Source::CheckPrettyRFailFull { compiler } |
            Source::CheckPrettyRPassValgrind { compiler } |
            Source::CheckRMake { compiler } => {
                vec![self.librustc(compiler),
                     self.target(compiler.host).tool_compiletest(compiler.stage)]
            }
            Source::CheckDocs { compiler } => {
                vec![self.libstd(compiler)]
            }
            Source::CheckErrorIndex { compiler } => {
                vec![self.libstd(compiler),
                     self.target(compiler.host).tool_error_index(compiler.stage)]
            }
            Source::CheckCrateStd { compiler } => {
                vec![self.libtest(compiler)]
            }
            Source::CheckCrateTest { compiler } => {
                vec![self.libtest(compiler)]
            }
            Source::CheckCrateRustc { compiler } => {
                vec![self.libtest(compiler)]
            }

            Source::ToolLinkchecker { stage } |
            Source::ToolTidy { stage } => {
                vec![self.libstd(self.compiler(stage))]
            }
            Source::ToolErrorIndex { stage } |
            Source::ToolRustbook { stage } => {
                vec![self.librustc(self.compiler(stage))]
            }
            Source::ToolCargoTest { stage } => {
                vec![self.libstd(self.compiler(stage))]
            }
            Source::ToolCompiletest { stage } => {
                vec![self.libtest(self.compiler(stage))]
            }

            Source::DistDocs { stage } => vec![self.doc(stage)],
            Source::DistMingw { _dummy: _ } => Vec::new(),
            Source::DistRustc { stage } => {
                vec![self.rustc(stage)]
            }
            Source::DistStd { compiler } => {
                // We want to package up as many target libraries as possible
                // for the `rust-std` package, so if this is a host target we
                // depend on librustc and otherwise we just depend on libtest.
                if build.config.host.iter().any(|t| t == self.target) {
                    vec![self.librustc(compiler)]
                } else {
                    vec![self.libtest(compiler)]
                }
            }
            Source::DistSrc { _dummy: _ } => Vec::new(),

            Source::Dist { stage } => {
                let mut base = Vec::new();

                for host in build.config.host.iter() {
                    let host = self.target(host);
                    base.push(host.dist_src(()));
                    base.push(host.dist_rustc(stage));
                    if host.target.contains("windows-gnu") {
                        base.push(host.dist_mingw(()));
                    }

                    let compiler = self.compiler(stage);
                    for target in build.config.target.iter() {
                        let target = self.target(target);
                        if build.config.docs {
                            base.push(target.dist_docs(stage));
                        }
                        base.push(target.dist_std(compiler));
                    }
                }
                return base
            }

            Source::AndroidCopyLibs { compiler } => {
                vec![self.libtest(compiler)]
            }
        }
    }
}
