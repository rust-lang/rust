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
            (rustc, Rustc { stage: u32 }),
            (libstd, Libstd { stage: u32, compiler: Compiler<'a> }),
            (librustc, Librustc { stage: u32, compiler: Compiler<'a> }),
            (llvm, Llvm { _dummy: () }),
            (compiler_rt, CompilerRt { _dummy: () }),
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
        for host in build.config.host.iter() {
            if !build.flags.host.contains(host) {
                continue
            }
            let host = t.target(host);
            targets.push(host.librustc(stage, host.compiler(stage)));
            for target in build.config.target.iter() {
                if !build.flags.target.contains(target) {
                    continue
                }
                targets.push(host.target(target)
                                 .libstd(stage, host.compiler(stage)));
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
    for step in build.flags.step.iter() {
        let compiler = host.compiler(stage);
        match &step[..] {
            "libstd" => targets.push(target.libstd(stage, compiler)),
            "librustc" => targets.push(target.libstd(stage, compiler)),
            "rustc" => targets.push(host.rustc(stage)),
            "llvm" => targets.push(target.llvm(())),
            "compiler-rt" => targets.push(target.compiler_rt(())),
            _ => panic!("unknown build target: `{}`", step),
        }
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
                if self.target == build.config.build {
                    Vec::new()
                } else {
                    let compiler = Compiler::new(0, &build.config.build);
                    vec![self.librustc(0, compiler)]
                }
            }
            Source::Rustc { stage } => {
                vec![self.librustc(stage - 1, self.compiler(stage - 1))]
            }
            Source::Librustc { stage, compiler } => {
                vec![self.libstd(stage, compiler), self.llvm(())]
            }
            Source::Libstd { stage: _, compiler } => {
                vec![self.compiler_rt(()),
                     self.rustc(compiler.stage).target(compiler.host)]
            }
            Source::CompilerRt { _dummy } => {
                vec![self.llvm(()).target(&build.config.build)]
            }
            Source::Llvm { _dummy } => Vec::new(),
        }
    }
}
