// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::HashSet;

use build::{Build, Compiler};

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct Step<'a> {
    pub src: Source<'a>,
    pub target: &'a str,
}

macro_rules! targets {
    ($m:ident) => {
        $m! {
            // Step representing building the stageN compiler. This is just the
            // compiler executable itself, not any of the support libraries
            (rustc, Rustc { stage: u32 }),

            // Steps for the two main cargo builds, one for the standard library
            // and one for the compiler itself. These are parameterized over the
            // stage output they're going to be placed in along with the
            // compiler which is producing the copy of libstd or librustc
            (libstd, Libstd { compiler: Compiler<'a> }),
            (librustc, Librustc { compiler: Compiler<'a> }),

            // Links the standard library/librustc produced by the compiler
            // provided into the host's directory also provided.
            (libstd_link, LibstdLink {
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

            // Steps for long-running native builds. Ideally these wouldn't
            // actually exist and would be part of build scripts, but for now
            // these are here.
            //
            // There aren't really any parameters to this, but empty structs
            // with braces are unstable so we just pick something that works.
            (llvm, Llvm { _dummy: () }),
            (compiler_rt, CompilerRt { _dummy: () }),

            // Steps for various pieces of documentation that we can generate,
            // the 'doc' step is just a pseudo target to depend on a bunch of
            // others.
            (doc, Doc { stage: u32 }),
            (doc_book, DocBook { stage: u32 }),
            (doc_nomicon, DocNomicon { stage: u32 }),
            (doc_style, DocStyle { stage: u32 }),
            (doc_standalone, DocStandalone { stage: u32 }),
            (doc_std, DocStd { stage: u32 }),
            (doc_rustc, DocRustc { stage: u32 }),
            (doc_error_index, DocErrorIndex { stage: u32 }),

            // Steps for running tests. The 'check' target is just a pseudo
            // target to depend on a bunch of others.
            (check, Check { stage: u32, compiler: Compiler<'a> }),
            (check_linkcheck, CheckLinkcheck { stage: u32 }),
        }
    }
}

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

    add_steps(build, stage, &host, &target, &mut targets);

    if targets.len() == 0 {
        let t = Step {
            src: Source::Llvm { _dummy: () },
            target: &build.config.build,
        };
        targets.push(t.doc(stage));
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
                                     .libstd(host.compiler(stage)));
                } else {
                    targets.push(host.target(target)
                                     .libstd_link(t.compiler(stage), host.target));
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

    targets!(constructors);

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
                vec![self.libstd(compiler), self.llvm(())]
            }
            Source::Libstd { compiler } => {
                vec![self.compiler_rt(()),
                     self.rustc(compiler.stage).target(compiler.host)]
            }
            Source::LibrustcLink { compiler, host } => {
                vec![self.librustc(compiler),
                     self.libstd_link(compiler, host)]
            }
            Source::LibstdLink { compiler, host } => {
                vec![self.libstd(compiler),
                     self.target(host).rustc(compiler.stage)]
            }
            Source::CompilerRt { _dummy } => {
                vec![self.llvm(()).target(&build.config.build)]
            }
            Source::Llvm { _dummy } => Vec::new(),
            Source::DocStd { stage } => {
                vec![self.libstd(self.compiler(stage))]
            }
            Source::DocBook { stage } |
            Source::DocNomicon { stage } |
            Source::DocStyle { stage } => {
                vec![self.tool_rustbook(stage)]
            }
            Source::DocErrorIndex { stage } => {
                vec![self.tool_error_index(stage)]
            }
            Source::DocStandalone { stage } => {
                vec![self.rustc(stage)]
            }
            Source::DocRustc { stage } => {
                vec![self.doc_std(stage)]
            }
            Source::Doc { stage } => {
                vec![self.doc_book(stage), self.doc_nomicon(stage),
                     self.doc_style(stage), self.doc_standalone(stage),
                     self.doc_std(stage),
                     self.doc_error_index(stage)]
            }
            Source::Check { stage, compiler: _ } => {
                vec![self.check_linkcheck(stage)]
            }
            Source::CheckLinkcheck { stage } => {
                vec![self.tool_linkchecker(stage), self.doc(stage)]
            }

            Source::ToolLinkchecker { stage } => {
                vec![self.libstd(self.compiler(stage))]
            }
            Source::ToolErrorIndex { stage } |
            Source::ToolRustbook { stage } => {
                vec![self.librustc(self.compiler(stage))]
            }
        }
    }
}
