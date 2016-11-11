// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::collections::{HashMap, HashSet};
use std::mem;

use check;
use compile;
use dist;
use doc;
use flags::Subcommand;
use install;
use native;
use {Compiler, Build, Mode};

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
struct Step<'a> {
    name: &'a str,
    stage: u32,
    host: &'a str,
    target: &'a str,
}

impl<'a> Step<'a> {
    fn name(&self, name: &'a str) -> Step<'a> {
        Step { name: name, ..*self }
    }

    fn stage(&self, stage: u32) -> Step<'a> {
        Step { stage: stage, ..*self }
    }

    fn host(&self, host: &'a str) -> Step<'a> {
        Step { host: host, ..*self }
    }

    fn target(&self, target: &'a str) -> Step<'a> {
        Step { target: target, ..*self }
    }

    fn compiler(&self) -> Compiler<'a> {
        Compiler::new(self.stage, self.host)
    }
}

pub fn run(build: &Build) {
    let rules = build_rules(build);
    let steps = rules.plan();
    rules.run(&steps);
}

pub fn build_rules(build: &Build) -> Rules {
    let mut rules: Rules = Rules::new(build);
    // dummy rule to do nothing, useful when a dep maps to no deps
    rules.build("dummy", "path/to/nowhere");
    fn dummy<'a>(s: &Step<'a>, build: &'a Build) -> Step<'a> {
        s.name("dummy").stage(0)
         .target(&build.config.build)
         .host(&build.config.build)
    }

    // Helper for loading an entire DAG of crates, rooted at `name`
    let krates = |name: &str| {
        let mut ret = Vec::new();
        let mut list = vec![name];
        let mut visited = HashSet::new();
        while let Some(krate) = list.pop() {
            let default = krate == name;
            let krate = &build.crates[krate];
            let path = krate.path.strip_prefix(&build.src).unwrap();
            ret.push((krate, path.to_str().unwrap(), default));
            for dep in krate.deps.iter() {
                if visited.insert(dep) && dep != "build_helper" {
                    list.push(dep);
                }
            }
        }
        return ret
    };

    rules.build("rustc", "path/to/nowhere")
         .dep(move |s| {
             if s.stage == 0 {
                 dummy(s, build)
             } else {
                 s.name("librustc")
                  .host(&build.config.build)
                  .stage(s.stage - 1)
             }
         })
         .run(move |s| compile::assemble_rustc(build, s.stage, s.target));
    rules.build("llvm", "src/llvm")
         .host(true)
         .run(move |s| native::llvm(build, s.target));

    // ========================================================================
    // Crate compilations
    //
    // Tools used during the build system but not shipped
    rules.build("libstd", "src/libstd")
         .dep(|s| s.name("build-crate-std_shim"));
    rules.build("libtest", "src/libtest")
         .dep(|s| s.name("build-crate-test_shim"));
    rules.build("librustc", "src/librustc")
         .dep(|s| s.name("build-crate-rustc-main"));
    for (krate, path, _default) in krates("std_shim") {
        rules.build(&krate.build_step, path)
             .dep(move |s| s.name("rustc").host(&build.config.build).target(s.host))
             .dep(move |s| {
                 if s.host == build.config.build {
                    dummy(s, build)
                 } else {
                    s.host(&build.config.build)
                 }
             })
             .run(move |s| {
                 if s.host == build.config.build {
                    compile::std(build, s.target, &s.compiler())
                 } else {
                    compile::std_link(build, s.target, s.stage, s.host)
                 }
             });
    }
    for (krate, path, default) in krates("test_shim") {
        rules.build(&krate.build_step, path)
             .dep(|s| s.name("libstd"))
             .dep(move |s| {
                 if s.host == build.config.build {
                    dummy(s, build)
                 } else {
                    s.host(&build.config.build)
                 }
             })
             .default(default)
             .run(move |s| {
                 if s.host == build.config.build {
                    compile::test(build, s.target, &s.compiler())
                 } else {
                    compile::test_link(build, s.target, s.stage, s.host)
                 }
             });
    }
    for (krate, path, default) in krates("rustc-main") {
        rules.build(&krate.build_step, path)
             .dep(|s| s.name("libtest"))
             .dep(move |s| s.name("llvm").host(&build.config.build).stage(0))
             .dep(move |s| {
                 if s.host == build.config.build {
                    dummy(s, build)
                 } else {
                    s.host(&build.config.build)
                 }
             })
             .host(true)
             .default(default)
             .run(move |s| {
                 if s.host == build.config.build {
                    compile::rustc(build, s.target, &s.compiler())
                 } else {
                    compile::rustc_link(build, s.target, s.stage, s.host)
                 }
             });
    }

    // ========================================================================
    // Test targets
    //
    // Various unit tests and tests suites we can run
    {
        let mut suite = |name, path, dir, mode| {
            rules.test(name, path)
                 .dep(|s| s.name("libtest"))
                 .dep(|s| s.name("tool-compiletest").target(s.host))
                 .dep(|s| s.name("test-helpers"))
                 .dep(move |s| {
                     if s.target.contains("android") {
                         s.name("android-copy-libs")
                     } else {
                         dummy(s, build)
                     }
                 })
                 .default(true)
                 .run(move |s| {
                     check::compiletest(build, &s.compiler(), s.target, dir, mode)
                 });
        };

        suite("check-rpass", "src/test/run-pass", "run-pass", "run-pass");
        suite("check-cfail", "src/test/compile-fail", "compile-fail", "compile-fail");
        suite("check-pfail", "src/test/parse-fail", "parse-fail", "parse-fail");
        suite("check-rfail", "src/test/run-fail", "run-fail", "run-fail");
        suite("check-rpass-valgrind", "src/test/run-pass-valgrind",
              "run-pass-valgrind", "run-pass-valgrind");
        suite("check-mir-opt", "src/test/mir-opt", "mir-opt", "mir-opt");
        if build.config.codegen_tests {
            suite("check-codegen", "src/test/codegen", "codegen", "codegen");
        }
        suite("check-codegen-units", "src/test/codegen-units", "codegen-units",
              "codegen-units");
        suite("check-incremental", "src/test/incremental", "incremental",
              "incremental");
        suite("check-ui", "src/test/ui", "ui", "ui");
        suite("check-pretty", "src/test/pretty", "pretty", "pretty");
        suite("check-pretty-rpass", "src/test/run-pass/pretty", "pretty",
              "run-pass");
        suite("check-pretty-rfail", "src/test/run-pass/pretty", "pretty",
              "run-fail");
        suite("check-pretty-valgrind", "src/test/run-pass-valgrind", "pretty",
              "run-pass-valgrind");
    }

    if build.config.build.contains("msvc") {
        // nothing to do for debuginfo tests
    } else if build.config.build.contains("apple") {
        rules.test("check-debuginfo", "src/test/debuginfo")
             .dep(|s| s.name("libtest"))
             .dep(|s| s.name("tool-compiletest").host(s.host))
             .dep(|s| s.name("test-helpers"))
             .dep(|s| s.name("debugger-scripts"))
             .run(move |s| check::compiletest(build, &s.compiler(), s.target,
                                         "debuginfo-lldb", "debuginfo"));
    } else {
        rules.test("check-debuginfo", "src/test/debuginfo")
             .dep(|s| s.name("libtest"))
             .dep(|s| s.name("tool-compiletest").host(s.host))
             .dep(|s| s.name("test-helpers"))
             .dep(|s| s.name("debugger-scripts"))
             .run(move |s| check::compiletest(build, &s.compiler(), s.target,
                                         "debuginfo-gdb", "debuginfo"));
    }

    rules.test("debugger-scripts", "src/etc/lldb_batchmode.py")
         .run(move |s| dist::debugger_scripts(build, &build.sysroot(&s.compiler()),
                                         s.target));

    {
        let mut suite = |name, path, dir, mode| {
            rules.test(name, path)
                 .dep(|s| s.name("librustc"))
                 .dep(|s| s.name("tool-compiletest").target(s.host))
                 .default(true)
                 .host(true)
                 .run(move |s| {
                     check::compiletest(build, &s.compiler(), s.target, dir, mode)
                 });
        };

        suite("check-rpass-full", "src/test/run-pass-fulldeps",
              "run-pass", "run-pass-fulldeps");
        suite("check-cfail-full", "src/test/compile-fail-fulldeps",
              "compile-fail", "compile-fail-fulldeps");
        suite("check-rmake", "src/test/run-make", "run-make", "run-make");
        suite("check-rustdoc", "src/test/rustdoc", "rustdoc", "rustdoc");
        suite("check-pretty-rpass-full", "src/test/run-pass-fulldeps",
              "pretty", "run-pass-fulldeps");
        suite("check-pretty-rfail-full", "src/test/run-fail-fulldeps",
              "pretty", "run-fail-fulldeps");
    }

    for (krate, path, _default) in krates("std_shim") {
        rules.test(&krate.test_step, path)
             .dep(|s| s.name("libtest"))
             .run(move |s| check::krate(build, &s.compiler(), s.target,
                                        Mode::Libstd, Some(&krate.name)));
    }
    rules.test("check-std-all", "path/to/nowhere")
         .dep(|s| s.name("libtest"))
         .default(true)
         .run(move |s| check::krate(build, &s.compiler(), s.target, Mode::Libstd,
                               None));
    for (krate, path, _default) in krates("test_shim") {
        rules.test(&krate.test_step, path)
             .dep(|s| s.name("libtest"))
             .run(move |s| check::krate(build, &s.compiler(), s.target,
                                        Mode::Libtest, Some(&krate.name)));
    }
    rules.test("check-test-all", "path/to/nowhere")
         .dep(|s| s.name("libtest"))
         .default(true)
         .run(move |s| check::krate(build, &s.compiler(), s.target, Mode::Libtest,
                               None));
    for (krate, path, _default) in krates("rustc-main") {
        rules.test(&krate.test_step, path)
             .dep(|s| s.name("librustc"))
             .host(true)
             .run(move |s| check::krate(build, &s.compiler(), s.target,
                                        Mode::Librustc, Some(&krate.name)));
    }
    rules.test("check-rustc-all", "path/to/nowhere")
         .dep(|s| s.name("librustc"))
         .default(true)
         .host(true)
         .run(move |s| check::krate(build, &s.compiler(), s.target, Mode::Librustc,
                               None));

    rules.test("check-linkchecker", "src/tools/linkchecker")
         .dep(|s| s.name("tool-linkchecker"))
         .dep(|s| s.name("default:doc"))
         .default(true)
         .host(true)
         .run(move |s| check::linkcheck(build, s.stage, s.target));
    rules.test("check-cargotest", "src/tools/cargotest")
         .dep(|s| s.name("tool-cargotest"))
         .dep(|s| s.name("librustc"))
         .host(true)
         .run(move |s| check::cargotest(build, s.stage, s.target));
    rules.test("check-tidy", "src/tools/tidy")
         .dep(|s| s.name("tool-tidy"))
         .default(true)
         .host(true)
         .run(move |s| check::tidy(build, s.stage, s.target));
    rules.test("check-error-index", "src/tools/error_index_generator")
         .dep(|s| s.name("libstd"))
         .dep(|s| s.name("tool-error-index").host(s.host))
         .default(true)
         .host(true)
         .run(move |s| check::error_index(build, &s.compiler()));
    rules.test("check-docs", "src/doc")
         .dep(|s| s.name("libtest"))
         .default(true)
         .host(true)
         .run(move |s| check::docs(build, &s.compiler()));

    rules.build("test-helpers", "src/rt/rust_test_helpers.c")
         .run(move |s| native::test_helpers(build, s.target));
    rules.test("android-copy-libs", "path/to/nowhere")
         .dep(|s| s.name("libtest"))
         .run(move |s| check::android_copy_libs(build, &s.compiler(), s.target));

    // ========================================================================
    // Build tools
    //
    // Tools used during the build system but not shipped
    rules.build("tool-rustbook", "src/tools/rustbook")
         .dep(|s| s.name("librustc"))
         .run(move |s| compile::tool(build, s.stage, s.target, "rustbook"));
    rules.build("tool-error-index", "src/tools/error_index_generator")
         .dep(|s| s.name("librustc"))
         .run(move |s| compile::tool(build, s.stage, s.target, "error_index_generator"));
    rules.build("tool-tidy", "src/tools/tidy")
         .dep(|s| s.name("libstd"))
         .run(move |s| compile::tool(build, s.stage, s.target, "tidy"));
    rules.build("tool-linkchecker", "src/tools/linkchecker")
         .dep(|s| s.name("libstd"))
         .run(move |s| compile::tool(build, s.stage, s.target, "linkchecker"));
    rules.build("tool-cargotest", "src/tools/cargotest")
         .dep(|s| s.name("libstd"))
         .run(move |s| compile::tool(build, s.stage, s.target, "cargotest"));
    rules.build("tool-compiletest", "src/tools/compiletest")
         .dep(|s| s.name("libtest"))
         .run(move |s| compile::tool(build, s.stage, s.target, "compiletest"));

    // ========================================================================
    // Documentation targets
    rules.doc("doc-book", "src/doc/book")
         .dep(move |s| s.name("tool-rustbook").target(&build.config.build))
         .default(build.config.docs)
         .run(move |s| doc::rustbook(build, s.stage, s.target, "book"));
    rules.doc("doc-nomicon", "src/doc/nomicon")
         .dep(move |s| s.name("tool-rustbook").target(&build.config.build))
         .default(build.config.docs)
         .run(move |s| doc::rustbook(build, s.stage, s.target, "nomicon"));
    rules.doc("doc-standalone", "src/doc")
         .dep(move |s| s.name("rustc").host(&build.config.build).target(&build.config.build))
         .default(build.config.docs)
         .run(move |s| doc::standalone(build, s.stage, s.target));
    rules.doc("doc-error-index", "src/tools/error_index_generator")
         .dep(move |s| s.name("tool-error-index").target(&build.config.build))
         .dep(move |s| s.name("librustc"))
         .default(build.config.docs)
         .host(true)
         .run(move |s| doc::error_index(build, s.stage, s.target));
    for (krate, path, default) in krates("std_shim") {
        rules.doc(&krate.doc_step, path)
             .dep(|s| s.name("libstd"))
             .default(default && build.config.docs)
             .run(move |s| doc::std(build, s.stage, s.target));
    }
    for (krate, path, default) in krates("test_shim") {
        rules.doc(&krate.doc_step, path)
             .dep(|s| s.name("libtest"))
             .default(default && build.config.docs)
             .run(move |s| doc::test(build, s.stage, s.target));
    }
    for (krate, path, default) in krates("rustc-main") {
        rules.doc(&krate.doc_step, path)
             .dep(|s| s.name("librustc"))
             .host(true)
             .default(default && build.config.compiler_docs)
             .run(move |s| doc::rustc(build, s.stage, s.target));
    }

    // ========================================================================
    // Distribution targets
    rules.dist("dist-rustc", "src/librustc")
         .dep(move |s| s.name("rustc").host(&build.config.build))
         .host(true)
         .default(true)
         .run(move |s| dist::rustc(build, s.stage, s.target));
    rules.dist("dist-std", "src/libstd")
         .dep(move |s| {
             // We want to package up as many target libraries as possible
             // for the `rust-std` package, so if this is a host target we
             // depend on librustc and otherwise we just depend on libtest.
             if build.config.host.iter().any(|t| t == s.target) {
                 s.name("librustc")
             } else {
                 s.name("libtest")
             }
         })
         .default(true)
         .run(move |s| dist::std(build, &s.compiler(), s.target));
    rules.dist("dist-mingw", "path/to/nowhere")
         .run(move |s| dist::mingw(build, s.target));
    rules.dist("dist-src", "src")
         .default(true)
         .host(true)
         .run(move |_| dist::rust_src(build));
    rules.dist("dist-docs", "src/doc")
         .default(true)
         .dep(|s| s.name("default:doc"))
         .run(move |s| dist::docs(build, s.stage, s.target));
    rules.dist("install", "src")
         .dep(|s| s.name("default:dist"))
         .run(move |s| install::install(build, s.stage, s.target));

    rules.verify();
    return rules
}

struct Rule<'a> {
    name: &'a str,
    path: &'a str,
    kind: Kind,
    deps: Vec<Box<Fn(&Step<'a>) -> Step<'a> + 'a>>,
    run: Box<Fn(&Step<'a>) + 'a>,
    default: bool,
    host: bool,
}

#[derive(PartialEq)]
enum Kind {
    Build,
    Test,
    Dist,
    Doc,
}

impl<'a> Rule<'a> {
    fn new(name: &'a str, path: &'a str, kind: Kind) -> Rule<'a> {
        Rule {
            name: name,
            deps: Vec::new(),
            run: Box::new(|_| ()),
            path: path,
            kind: kind,
            default: false,
            host: false,
        }
    }
}

struct RuleBuilder<'a: 'b, 'b> {
    rules: &'b mut Rules<'a>,
    rule: Rule<'a>,
}

impl<'a, 'b> RuleBuilder<'a, 'b> {
    fn dep<F>(&mut self, f: F) -> &mut Self
        where F: Fn(&Step<'a>) -> Step<'a> + 'a,
    {
        self.rule.deps.push(Box::new(f));
        self
    }

    fn run<F>(&mut self, f: F) -> &mut Self
        where F: Fn(&Step<'a>) + 'a,
    {
        self.rule.run = Box::new(f);
        self
    }

    fn default(&mut self, default: bool) -> &mut Self {
        self.rule.default = default;
        self
    }

    fn host(&mut self, host: bool) -> &mut Self {
        self.rule.host = host;
        self
    }
}

impl<'a, 'b> Drop for RuleBuilder<'a, 'b> {
    fn drop(&mut self) {
        let rule = mem::replace(&mut self.rule, Rule::new("", "", Kind::Build));
        let prev = self.rules.rules.insert(rule.name, rule);
        if let Some(prev) = prev {
            panic!("duplicate rule named: {}", prev.name);
        }
    }
}

pub struct Rules<'a> {
    build: &'a Build,
    sbuild: Step<'a>,
    rules: HashMap<&'a str, Rule<'a>>,
}

impl<'a> Rules<'a> {
    fn new(build: &'a Build) -> Rules<'a> {
        Rules {
            build: build,
            sbuild: Step {
                stage: build.flags.stage.unwrap_or(2),
                target: &build.config.build,
                host: &build.config.build,
                name: "",
            },
            rules: HashMap::new(),
        }
    }

    fn build<'b>(&'b mut self, name: &'a str, path: &'a str)
                 -> RuleBuilder<'a, 'b> {
        self.rule(name, path, Kind::Build)
    }

    fn test<'b>(&'b mut self, name: &'a str, path: &'a str)
                -> RuleBuilder<'a, 'b> {
        self.rule(name, path, Kind::Test)
    }

    fn doc<'b>(&'b mut self, name: &'a str, path: &'a str)
               -> RuleBuilder<'a, 'b> {
        self.rule(name, path, Kind::Doc)
    }

    fn dist<'b>(&'b mut self, name: &'a str, path: &'a str)
                -> RuleBuilder<'a, 'b> {
        self.rule(name, path, Kind::Dist)
    }

    fn rule<'b>(&'b mut self,
                name: &'a str,
                path: &'a str,
                kind: Kind) -> RuleBuilder<'a, 'b> {
        RuleBuilder {
            rules: self,
            rule: Rule::new(name, path, kind),
        }
    }

    /// Verify the dependency graph defined by all our rules are correct, e.g.
    /// everything points to a valid something else.
    fn verify(&self) {
        for rule in self.rules.values() {
            for dep in rule.deps.iter() {
                let dep = dep(&self.sbuild.name(rule.name));
                if self.rules.contains_key(&dep.name) || dep.name.starts_with("default:") {
                    continue }
                panic!("\

invalid rule dependency graph detected, was a rule added and maybe typo'd?

    `{}` depends on `{}` which does not exist

", rule.name, dep.name);
            }
        }
    }

    pub fn print_help(&self, command: &str) {
        let kind = match command {
            "build" => Kind::Build,
            "doc" => Kind::Doc,
            "test" => Kind::Test,
            "dist" => Kind::Dist,
            _ => return,
        };
        let rules = self.rules.values().filter(|r| r.kind == kind);
        let rules = rules.filter(|r| !r.path.contains("nowhere"));
        let mut rules = rules.collect::<Vec<_>>();
        rules.sort_by_key(|r| r.path);

        println!("Available paths:\n");
        for rule in rules {
            print!("    ./x.py {} {}", command, rule.path);

            println!("");
        }
    }

    /// Construct the top-level build steps that we're going to be executing,
    /// given the subcommand that our build is performing.
    fn plan(&self) -> Vec<Step<'a>> {
        let (kind, paths) = match self.build.flags.cmd {
            Subcommand::Build { ref paths } => (Kind::Build, &paths[..]),
            Subcommand::Doc { ref paths } => (Kind::Doc, &paths[..]),
            Subcommand::Test { ref paths, test_args: _ } => (Kind::Test, &paths[..]),
            Subcommand::Dist { install } => {
                if install {
                    return vec![self.sbuild.name("install")]
                } else {
                    (Kind::Dist, &[][..])
                }
            }
            Subcommand::Clean => panic!(),
        };

        self.rules.values().filter(|rule| rule.kind == kind).filter(|rule| {
            (paths.len() == 0 && rule.default) || paths.iter().any(|path| {
                path.ends_with(rule.path)
            })
        }).flat_map(|rule| {
            let hosts = if self.build.flags.host.len() > 0 {
                &self.build.flags.host
            } else {
                &self.build.config.host
            };
            let targets = if self.build.flags.target.len() > 0 {
                &self.build.flags.target
            } else {
                &self.build.config.target
            };
            let arr = if rule.host {hosts} else {targets};

            hosts.iter().flat_map(move |host| {
                arr.iter().map(move |target| {
                    self.sbuild.name(rule.name).target(target).host(host)
                })
            })
        }).collect()
    }

    /// Execute all top-level targets indicated by `steps`.
    ///
    /// This will take the list returned by `plan` and then execute each step
    /// along with all required dependencies as it goes up the chain.
    fn run(&self, steps: &[Step<'a>]) {
        self.build.verbose("bootstrap top targets:");
        for step in steps.iter() {
            self.build.verbose(&format!("\t{:?}", step));
        }

        // Using `steps` as the top-level targets, make a topological ordering
        // of what we need to do.
        let mut order = Vec::new();
        let mut added = HashSet::new();
        for step in steps.iter().cloned() {
            self.fill(step, &mut order, &mut added);
        }

        // Print out what we're doing for debugging
        self.build.verbose("bootstrap build plan:");
        for step in order.iter() {
            self.build.verbose(&format!("\t{:?}", step));
        }

        // And finally, iterate over everything and execute it.
        for step in order.iter() {
            self.build.verbose(&format!("executing step {:?}", step));
            (self.rules[step.name].run)(step);
        }
    }

    fn fill(&self,
            step: Step<'a>,
            order: &mut Vec<Step<'a>>,
            added: &mut HashSet<Step<'a>>) {
        if !added.insert(step.clone()) {
            return
        }
        for dep in self.rules[step.name].deps.iter() {
            let dep = dep(&step);
            if dep.name.starts_with("default:") {
                let kind = match &dep.name[8..] {
                    "doc" => Kind::Doc,
                    "dist" => Kind::Dist,
                    kind => panic!("unknown kind: `{}`", kind),
                };
                let rules = self.rules.values().filter(|r| r.default);
                for rule in rules.filter(|r| r.kind == kind) {
                    self.fill(dep.name(rule.name), order, added);
                }
            } else {
                self.fill(dep, order, added);
            }
        }
        order.push(step);
    }
}
