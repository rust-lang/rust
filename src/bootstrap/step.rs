// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Definition of steps of the build system.
//!
//! This is where some of the real meat of rustbuild is located, in how we
//! define targets and the dependencies amongst them. This file can sort of be
//! viewed as just defining targets in a makefile which shell out to predefined
//! functions elsewhere about how to execute the target.
//!
//! The primary function here you're likely interested in is the `build_rules`
//! function. This will create a `Rules` structure which basically just lists
//! everything that rustbuild can do. Each rule has a human-readable name, a
//! path associated with it, some dependencies, and then a closure of how to
//! actually perform the rule.
//!
//! All steps below are defined in self-contained units, so adding a new target
//! to the build system should just involve adding the meta information here
//! along with the actual implementation elsewhere. You can find more comments
//! about how to define rules themselves below.

use std::collections::{HashMap, HashSet};
use std::mem;

use check::{self, TestKind};
use compile;
use dist;
use doc;
use flags::Subcommand;
use install;
use native;
use {Compiler, Build, Mode};

pub fn run(build: &Build) {
    let rules = build_rules(build);
    let steps = rules.plan();
    rules.run(&steps);
}

pub fn build_rules<'a>(build: &'a Build) -> Rules {
    let mut rules = Rules::new(build);

    // This is the first rule that we're going to define for rustbuild, which is
    // used to compile LLVM itself. All rules are added through the `rules`
    // structure created above and are configured through a builder-style
    // interface.
    //
    // First up we see the `build` method. This represents a rule that's part of
    // the top-level `build` subcommand. For example `./x.py build` is what this
    // is associating with. Note that this is normally only relevant if you flag
    // a rule as `default`, which we'll talk about later.
    //
    // Next up we'll see two arguments to this method:
    //
    // * `llvm` - this is the "human readable" name of this target. This name is
    //            not accessed anywhere outside this file itself (e.g. not in
    //            the CLI nor elsewhere in rustbuild). The purpose of this is to
    //            easily define dependencies between rules. That is, other rules
    //            will depend on this with the name "llvm".
    // * `src/llvm` - this is the relevant path to the rule that we're working
    //                with. This path is the engine behind how commands like
    //                `./x.py build src/llvm` work. This should typically point
    //                to the relevant component, but if there's not really a
    //                path to be assigned here you can pass something like
    //                `path/to/nowhere` to ignore it.
    //
    // After we create the rule with the `build` method we can then configure
    // various aspects of it. For example this LLVM rule uses `.host(true)` to
    // flag that it's a rule only for host targets. In other words, LLVM isn't
    // compiled for targets configured through `--target` (e.g. those we're just
    // building a standard library for).
    //
    // Next up the `dep` method will add a dependency to this rule. The closure
    // is yielded the step that represents executing the `llvm` rule itself
    // (containing information like stage, host, target, ...) and then it must
    // return a target that the step depends on. Here LLVM is actually
    // interesting where a cross-compiled LLVM depends on the host LLVM, but
    // otherwise it has no dependencies.
    //
    // To handle this we do a bit of dynamic dispatch to see what the dependency
    // is. If we're building a LLVM for the build triple, then we don't actually
    // have any dependencies! To do that we return a dependency on the `Step::noop()`
    // target which does nothing.
    //
    // If we're build a cross-compiled LLVM, however, we need to assemble the
    // libraries from the previous compiler. This step has the same name as
    // ours (llvm) but we want it for a different target, so we use the
    // builder-style methods on `Step` to configure this target to the build
    // triple.
    //
    // Finally, to finish off this rule, we define how to actually execute it.
    // That logic is all defined in the `native` module so we just delegate to
    // the relevant function there. The argument to the closure passed to `run`
    // is a `Step` (defined below) which encapsulates information like the
    // stage, target, host, etc.
    rules.build("llvm", "src/llvm")
         .host(true)
         .dep(move |s| {
             if s.target == build.config.build {
                 Step::noop()
             } else {
                 s.target(&build.config.build)
             }
         })
         .run(move |s| native::llvm(build, s.target));

    // Ok! After that example rule  that's hopefully enough to explain what's
    // going on here. You can check out the API docs below and also see a bunch
    // more examples of rules directly below as well.

    // the compiler with no target libraries ready to go
    rules.build("rustc", "src/rustc")
         .dep(|s| s.name("create-sysroot").target(s.host))
         .dep(move |s| {
             if s.stage == 0 {
                 Step::noop()
             } else {
                 s.name("librustc")
                  .host(&build.config.build)
                  .stage(s.stage - 1)
             }
         })
         .run(move |s| compile::assemble_rustc(build, s.stage, s.target));

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

    // ========================================================================
    // Crate compilations
    //
    // Tools used during the build system but not shipped
    rules.build("create-sysroot", "path/to/nowhere")
         .run(move |s| compile::create_sysroot(build, &s.compiler()));

    // These rules are "pseudo rules" that don't actually do any work
    // themselves, but represent a complete sysroot with the relevant compiler
    // linked into place.
    //
    // That is, depending on "libstd" means that when the rule is completed then
    // the `stage` sysroot for the compiler `host` will be available with a
    // standard library built for `target` linked in place. Not all rules need
    // the compiler itself to be available, just the standard library, so
    // there's a distinction between the two.
    rules.build("libstd", "src/libstd")
         .dep(|s| s.name("rustc").target(s.host))
         .dep(|s| s.name("libstd-link"));
    rules.build("libtest", "src/libtest")
         .dep(|s| s.name("libstd"))
         .dep(|s| s.name("libtest-link"))
         .default(true);
    rules.build("librustc", "src/librustc")
         .dep(|s| s.name("libtest"))
         .dep(|s| s.name("librustc-link"))
         .host(true)
         .default(true);

    // Helper method to define the rules to link a crate into its place in the
    // sysroot.
    //
    // The logic here is a little subtle as there's a few cases to consider.
    // Not all combinations of (stage, host, target) actually require something
    // to be compiled, but rather libraries could get propagated from a
    // different location. For example:
    //
    // * Any crate with a `host` that's not the build triple will not actually
    //   compile something. A different `host` means that the build triple will
    //   actually compile the libraries, and then we'll copy them over from the
    //   build triple to the `host` directory.
    //
    // * Some crates aren't even compiled by the build triple, but may be copied
    //   from previous stages. For example if we're not doing a full bootstrap
    //   then we may just depend on the stage1 versions of libraries to be
    //   available to get linked forward.
    //
    // * Finally, there are some cases, however, which do indeed comiple crates
    //   and link them into place afterwards.
    //
    // The rule definition below mirrors these three cases. The `dep` method
    // calculates the correct dependency which either comes from stage1, a
    // different compiler, or from actually building the crate itself (the `dep`
    // rule). The `run` rule then mirrors these three cases and links the cases
    // forward into the compiler sysroot specified from the correct location.
    fn crate_rule<'a, 'b>(build: &'a Build,
                          rules: &'b mut Rules<'a>,
                          krate: &'a str,
                          dep: &'a str,
                          link: fn(&Build, &Compiler, &Compiler, &str))
                          -> RuleBuilder<'a, 'b> {
        let mut rule = rules.build(&krate, "path/to/nowhere");
        rule.dep(move |s| {
                if build.force_use_stage1(&s.compiler(), s.target) {
                    s.host(&build.config.build).stage(1)
                } else if s.host == build.config.build {
                    s.name(dep)
                } else {
                    s.host(&build.config.build)
                }
            })
            .run(move |s| {
                if build.force_use_stage1(&s.compiler(), s.target) {
                    link(build,
                         &s.stage(1).host(&build.config.build).compiler(),
                         &s.compiler(),
                         s.target)
                } else if s.host == build.config.build {
                    link(build, &s.compiler(), &s.compiler(), s.target)
                } else {
                    link(build,
                         &s.host(&build.config.build).compiler(),
                         &s.compiler(),
                         s.target)
                }
            });
            return rule
    }

    // Similar to the `libstd`, `libtest`, and `librustc` rules above, except
    // these rules only represent the libraries being available in the sysroot,
    // not the compiler itself. This is done as not all rules need a compiler in
    // the sysroot, but may just need the libraries.
    //
    // All of these rules use the helper definition above.
    crate_rule(build,
               &mut rules,
               "libstd-link",
               "build-crate-std_shim",
               compile::std_link)
        .dep(|s| s.name("startup-objects"))
        .dep(|s| s.name("create-sysroot").target(s.host));
    crate_rule(build,
               &mut rules,
               "libtest-link",
               "build-crate-test_shim",
               compile::test_link)
        .dep(|s| s.name("libstd-link"));
    crate_rule(build,
               &mut rules,
               "librustc-link",
               "build-crate-rustc-main",
               compile::rustc_link)
        .dep(|s| s.name("libtest-link"));

    for (krate, path, _default) in krates("std_shim") {
        rules.build(&krate.build_step, path)
             .dep(|s| s.name("startup-objects"))
             .dep(move |s| s.name("rustc").host(&build.config.build).target(s.host))
             .run(move |s| compile::std(build, s.target, &s.compiler()));
    }
    for (krate, path, _default) in krates("test_shim") {
        rules.build(&krate.build_step, path)
             .dep(|s| s.name("libstd-link"))
             .run(move |s| compile::test(build, s.target, &s.compiler()));
    }
    for (krate, path, _default) in krates("rustc-main") {
        rules.build(&krate.build_step, path)
             .dep(|s| s.name("libtest-link"))
             .dep(move |s| s.name("llvm").host(&build.config.build).stage(0))
             .run(move |s| compile::rustc(build, s.target, &s.compiler()));
    }

    rules.build("startup-objects", "src/rtstartup")
         .dep(|s| s.name("create-sysroot").target(s.host))
         .run(move |s| compile::build_startup_objects(build, &s.compiler(), s.target));

    // ========================================================================
    // Test targets
    //
    // Various unit tests and tests suites we can run
    {
        let mut suite = |name, path, mode, dir| {
            rules.test(name, path)
                 .dep(|s| s.name("libtest"))
                 .dep(|s| s.name("tool-compiletest").target(s.host).stage(0))
                 .dep(|s| s.name("test-helpers"))
                 .dep(|s| s.name("android-copy-libs"))
                 .default(mode != "pretty") // pretty tests don't run everywhere
                 .run(move |s| {
                     check::compiletest(build, &s.compiler(), s.target, mode, dir)
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
    }

    if build.config.build.contains("msvc") {
        // nothing to do for debuginfo tests
    } else {
        rules.test("check-debuginfo-lldb", "src/test/debuginfo-lldb")
             .dep(|s| s.name("libtest"))
             .dep(|s| s.name("tool-compiletest").target(s.host).stage(0))
             .dep(|s| s.name("test-helpers"))
             .dep(|s| s.name("debugger-scripts"))
             .run(move |s| check::compiletest(build, &s.compiler(), s.target,
                                         "debuginfo-lldb", "debuginfo"));
        rules.test("check-debuginfo-gdb", "src/test/debuginfo-gdb")
             .dep(|s| s.name("libtest"))
             .dep(|s| s.name("tool-compiletest").target(s.host).stage(0))
             .dep(|s| s.name("test-helpers"))
             .dep(|s| s.name("debugger-scripts"))
             .dep(|s| s.name("android-copy-libs"))
             .run(move |s| check::compiletest(build, &s.compiler(), s.target,
                                         "debuginfo-gdb", "debuginfo"));
        let mut rule = rules.test("check-debuginfo", "src/test/debuginfo");
        rule.default(true);
        if build.config.build.contains("apple") {
            rule.dep(|s| s.name("check-debuginfo-lldb"));
        } else {
            rule.dep(|s| s.name("check-debuginfo-gdb"));
        }
    }

    rules.test("debugger-scripts", "src/etc/lldb_batchmode.py")
         .run(move |s| dist::debugger_scripts(build, &build.sysroot(&s.compiler()),
                                         s.target));

    {
        let mut suite = |name, path, mode, dir| {
            rules.test(name, path)
                 .dep(|s| s.name("librustc"))
                 .dep(|s| s.name("test-helpers"))
                 .dep(|s| s.name("tool-compiletest").target(s.host).stage(0))
                 .default(mode != "pretty")
                 .host(true)
                 .run(move |s| {
                     check::compiletest(build, &s.compiler(), s.target, mode, dir)
                 });
        };

        suite("check-ui", "src/test/ui", "ui", "ui");
        suite("check-rpass-full", "src/test/run-pass-fulldeps",
              "run-pass", "run-pass-fulldeps");
        suite("check-rfail-full", "src/test/run-fail-fulldeps",
              "run-fail", "run-fail-fulldeps");
        suite("check-cfail-full", "src/test/compile-fail-fulldeps",
              "compile-fail", "compile-fail-fulldeps");
        suite("check-rmake", "src/test/run-make", "run-make", "run-make");
        suite("check-rustdoc", "src/test/rustdoc", "rustdoc", "rustdoc");
        suite("check-pretty", "src/test/pretty", "pretty", "pretty");
        suite("check-pretty-rpass", "src/test/run-pass/pretty", "pretty",
              "run-pass");
        suite("check-pretty-rfail", "src/test/run-fail/pretty", "pretty",
              "run-fail");
        suite("check-pretty-valgrind", "src/test/run-pass-valgrind/pretty", "pretty",
              "run-pass-valgrind");
        suite("check-pretty-rpass-full", "src/test/run-pass-fulldeps/pretty",
              "pretty", "run-pass-fulldeps");
        suite("check-pretty-rfail-full", "src/test/run-fail-fulldeps/pretty",
              "pretty", "run-fail-fulldeps");
    }

    for (krate, path, _default) in krates("std_shim") {
        rules.test(&krate.test_step, path)
             .dep(|s| s.name("libtest"))
             .dep(|s| s.name("android-copy-libs"))
             .run(move |s| check::krate(build, &s.compiler(), s.target,
                                        Mode::Libstd, TestKind::Test,
                                        Some(&krate.name)));
    }
    rules.test("check-std-all", "path/to/nowhere")
         .dep(|s| s.name("libtest"))
         .dep(|s| s.name("android-copy-libs"))
         .default(true)
         .run(move |s| check::krate(build, &s.compiler(), s.target,
                                    Mode::Libstd, TestKind::Test, None));

    // std benchmarks
    for (krate, path, _default) in krates("std_shim") {
        rules.bench(&krate.bench_step, path)
             .dep(|s| s.name("libtest"))
             .dep(|s| s.name("android-copy-libs"))
             .run(move |s| check::krate(build, &s.compiler(), s.target,
                                        Mode::Libstd, TestKind::Bench,
                                        Some(&krate.name)));
    }
    rules.bench("bench-std-all", "path/to/nowhere")
         .dep(|s| s.name("libtest"))
         .dep(|s| s.name("android-copy-libs"))
         .default(true)
         .run(move |s| check::krate(build, &s.compiler(), s.target,
                                    Mode::Libstd, TestKind::Bench, None));

    for (krate, path, _default) in krates("test_shim") {
        rules.test(&krate.test_step, path)
             .dep(|s| s.name("libtest"))
             .dep(|s| s.name("android-copy-libs"))
             .run(move |s| check::krate(build, &s.compiler(), s.target,
                                        Mode::Libtest, TestKind::Test,
                                        Some(&krate.name)));
    }
    rules.test("check-test-all", "path/to/nowhere")
         .dep(|s| s.name("libtest"))
         .dep(|s| s.name("android-copy-libs"))
         .default(true)
         .run(move |s| check::krate(build, &s.compiler(), s.target,
                                    Mode::Libtest, TestKind::Test, None));
    for (krate, path, _default) in krates("rustc-main") {
        rules.test(&krate.test_step, path)
             .dep(|s| s.name("librustc"))
             .dep(|s| s.name("android-copy-libs"))
             .host(true)
             .run(move |s| check::krate(build, &s.compiler(), s.target,
                                        Mode::Librustc, TestKind::Test,
                                        Some(&krate.name)));
    }
    rules.test("check-rustc-all", "path/to/nowhere")
         .dep(|s| s.name("librustc"))
         .dep(|s| s.name("android-copy-libs"))
         .default(true)
         .host(true)
         .run(move |s| check::krate(build, &s.compiler(), s.target,
                                    Mode::Librustc, TestKind::Test, None));

    rules.test("check-linkchecker", "src/tools/linkchecker")
         .dep(|s| s.name("tool-linkchecker").stage(0))
         .dep(|s| s.name("default:doc"))
         .default(true)
         .host(true)
         .run(move |s| check::linkcheck(build, s.target));
    rules.test("check-cargotest", "src/tools/cargotest")
         .dep(|s| s.name("tool-cargotest").stage(0))
         .dep(|s| s.name("librustc"))
         .host(true)
         .run(move |s| check::cargotest(build, s.stage, s.target));
    rules.test("check-tidy", "src/tools/tidy")
         .dep(|s| s.name("tool-tidy").stage(0))
         .default(true)
         .host(true)
         .only_build(true)
         .run(move |s| check::tidy(build, s.target));
    rules.test("check-error-index", "src/tools/error_index_generator")
         .dep(|s| s.name("libstd"))
         .dep(|s| s.name("tool-error-index").host(s.host).stage(0))
         .default(true)
         .host(true)
         .run(move |s| check::error_index(build, &s.compiler()));
    rules.test("check-docs", "src/doc")
         .dep(|s| s.name("libtest"))
         .default(true)
         .host(true)
         .run(move |s| check::docs(build, &s.compiler()));
    rules.test("check-distcheck", "distcheck")
         .dep(|s| s.name("dist-src"))
         .run(move |_| check::distcheck(build));


    rules.build("test-helpers", "src/rt/rust_test_helpers.c")
         .run(move |s| native::test_helpers(build, s.target));
    rules.test("android-copy-libs", "path/to/nowhere")
         .dep(|s| s.name("libtest"))
         .run(move |s| check::android_copy_libs(build, &s.compiler(), s.target));

    rules.test("check-bootstrap", "src/bootstrap")
         .default(true)
         .host(true)
         .only_build(true)
         .run(move |_| check::bootstrap(build));

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
         .dep(move |s| {
             s.name("tool-rustbook")
              .host(&build.config.build)
              .target(&build.config.build)
              .stage(0)
         })
         .default(build.config.docs)
         .run(move |s| doc::rustbook(build, s.target, "book"));
    rules.doc("doc-nomicon", "src/doc/nomicon")
         .dep(move |s| {
             s.name("tool-rustbook")
              .host(&build.config.build)
              .target(&build.config.build)
              .stage(0)
         })
         .default(build.config.docs)
         .run(move |s| doc::rustbook(build, s.target, "nomicon"));
    rules.doc("doc-standalone", "src/doc")
         .dep(move |s| {
             s.name("rustc")
              .host(&build.config.build)
              .target(&build.config.build)
              .stage(0)
         })
         .default(build.config.docs)
         .run(move |s| doc::standalone(build, s.target));
    rules.doc("doc-error-index", "src/tools/error_index_generator")
         .dep(move |s| s.name("tool-error-index").target(&build.config.build).stage(0))
         .dep(move |s| s.name("librustc-link"))
         .default(build.config.docs)
         .host(true)
         .run(move |s| doc::error_index(build, s.target));
    for (krate, path, default) in krates("std_shim") {
        rules.doc(&krate.doc_step, path)
             .dep(|s| s.name("libstd-link"))
             .default(default && build.config.docs)
             .run(move |s| doc::std(build, s.stage, s.target));
    }
    for (krate, path, default) in krates("test_shim") {
        rules.doc(&krate.doc_step, path)
             .dep(|s| s.name("libtest-link"))
             .default(default && build.config.compiler_docs)
             .run(move |s| doc::test(build, s.stage, s.target));
    }
    for (krate, path, default) in krates("rustc-main") {
        rules.doc(&krate.doc_step, path)
             .dep(|s| s.name("librustc-link"))
             .host(true)
             .default(default && build.config.compiler_docs)
             .run(move |s| doc::rustc(build, s.stage, s.target));
    }

    // ========================================================================
    // Distribution targets
    rules.dist("dist-rustc", "src/librustc")
         .dep(move |s| s.name("rustc").host(&build.config.build))
         .host(true)
         .only_host_build(true)
         .default(true)
         .run(move |s| dist::rustc(build, s.stage, s.target));
    rules.dist("dist-std", "src/libstd")
         .dep(move |s| {
             // We want to package up as many target libraries as possible
             // for the `rust-std` package, so if this is a host target we
             // depend on librustc and otherwise we just depend on libtest.
             if build.config.host.iter().any(|t| t == s.target) {
                 s.name("librustc-link")
             } else {
                 s.name("libtest-link")
             }
         })
         .default(true)
         .only_host_build(true)
         .run(move |s| dist::std(build, &s.compiler(), s.target));
    rules.dist("dist-mingw", "path/to/nowhere")
         .default(true)
         .only_host_build(true)
         .run(move |s| {
             if s.target.contains("pc-windows-gnu") {
                 dist::mingw(build, s.target)
             }
         });
    rules.dist("dist-src", "src")
         .default(true)
         .host(true)
         .only_build(true)
         .only_host_build(true)
         .run(move |_| dist::rust_src(build));
    rules.dist("dist-docs", "src/doc")
         .default(true)
         .only_host_build(true)
         .dep(|s| s.name("default:doc"))
         .run(move |s| dist::docs(build, s.stage, s.target));
    rules.dist("dist-analysis", "analysis")
         .dep(|s| s.name("dist-std"))
         .default(true)
         .only_host_build(true)
         .run(move |s| dist::analysis(build, &s.compiler(), s.target));
    rules.dist("install", "path/to/nowhere")
         .dep(|s| s.name("default:dist"))
         .run(move |s| install::install(build, s.stage, s.target));

    rules.verify();
    return rules;
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
struct Step<'a> {
    /// Human readable name of the rule this step is executing. Possible names
    /// are all defined above in `build_rules`.
    name: &'a str,

    /// The stage this step is executing in. This is typically 0, 1, or 2.
    stage: u32,

    /// This step will likely involve a compiler, and the target that compiler
    /// itself is built for is called the host, this variable. Typically this is
    /// the target of the build machine itself.
    host: &'a str,

    /// The target that this step represents generating. If you're building a
    /// standard library for a new suite of targets, for example, this'll be set
    /// to those targets.
    target: &'a str,
}

impl<'a> Step<'a> {
    fn noop() -> Step<'a> {
        Step { name: "", stage: 0, host: "", target: "" }
    }

    /// Creates a new step which is the same as this, except has a new name.
    fn name(&self, name: &'a str) -> Step<'a> {
        Step { name: name, ..*self }
    }

    /// Creates a new step which is the same as this, except has a new stage.
    fn stage(&self, stage: u32) -> Step<'a> {
        Step { stage: stage, ..*self }
    }

    /// Creates a new step which is the same as this, except has a new host.
    fn host(&self, host: &'a str) -> Step<'a> {
        Step { host: host, ..*self }
    }

    /// Creates a new step which is the same as this, except has a new target.
    fn target(&self, target: &'a str) -> Step<'a> {
        Step { target: target, ..*self }
    }

    /// Returns the `Compiler` structure that this step corresponds to.
    fn compiler(&self) -> Compiler<'a> {
        Compiler::new(self.stage, self.host)
    }
}

struct Rule<'a> {
    /// The human readable name of this target, defined in `build_rules`.
    name: &'a str,

    /// The path associated with this target, used in the `./x.py` driver for
    /// easy and ergonomic specification of what to do.
    path: &'a str,

    /// The "kind" of top-level command that this rule is associated with, only
    /// relevant if this is a default rule.
    kind: Kind,

    /// List of dependencies this rule has. Each dependency is a function from a
    /// step that's being executed to another step that should be executed.
    deps: Vec<Box<Fn(&Step<'a>) -> Step<'a> + 'a>>,

    /// How to actually execute this rule. Takes a step with contextual
    /// information and then executes it.
    run: Box<Fn(&Step<'a>) + 'a>,

    /// Whether or not this is a "default" rule. That basically means that if
    /// you run, for example, `./x.py test` whether it's included or not.
    default: bool,

    /// Whether or not this is a "host" rule, or in other words whether this is
    /// only intended for compiler hosts and not for targets that are being
    /// generated.
    host: bool,

    /// Whether this rule is only for steps where the host is the build triple,
    /// not anything in hosts or targets.
    only_host_build: bool,

    /// Whether this rule is only for the build triple, not anything in hosts or
    /// targets.
    only_build: bool,
}

#[derive(PartialEq)]
enum Kind {
    Build,
    Test,
    Bench,
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
            only_host_build: false,
            only_build: false,
        }
    }
}

/// Builder pattern returned from the various methods on `Rules` which will add
/// the rule to the internal list on `Drop`.
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

    fn only_build(&mut self, only_build: bool) -> &mut Self {
        self.rule.only_build = only_build;
        self
    }

    fn only_host_build(&mut self, only_host_build: bool) -> &mut Self {
        self.rule.only_host_build = only_host_build;
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

    /// Creates a new rule of `Kind::Build` with the specified human readable
    /// name and path associated with it.
    ///
    /// The builder returned should be configured further with information such
    /// as how to actually run this rule.
    fn build<'b>(&'b mut self, name: &'a str, path: &'a str)
                 -> RuleBuilder<'a, 'b> {
        self.rule(name, path, Kind::Build)
    }

    /// Same as `build`, but for `Kind::Test`.
    fn test<'b>(&'b mut self, name: &'a str, path: &'a str)
                -> RuleBuilder<'a, 'b> {
        self.rule(name, path, Kind::Test)
    }

    /// Same as `build`, but for `Kind::Bench`.
    fn bench<'b>(&'b mut self, name: &'a str, path: &'a str)
                -> RuleBuilder<'a, 'b> {
        self.rule(name, path, Kind::Bench)
    }

    /// Same as `build`, but for `Kind::Doc`.
    fn doc<'b>(&'b mut self, name: &'a str, path: &'a str)
               -> RuleBuilder<'a, 'b> {
        self.rule(name, path, Kind::Doc)
    }

    /// Same as `build`, but for `Kind::Dist`.
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
                    continue
                }
                if dep == Step::noop() {
                    continue
                }
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
            "bench" => Kind::Bench,
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
        // Ok, the logic here is pretty subtle, and involves quite a few
        // conditionals. The basic idea here is to:
        //
        // 1. First, filter all our rules to the relevant ones. This means that
        //    the command specified corresponds to one of our `Kind` variants,
        //    and we filter all rules based on that.
        //
        // 2. Next, we determine which rules we're actually executing. If a
        //    number of path filters were specified on the command line we look
        //    for those, otherwise we look for anything tagged `default`.
        //
        // 3. Finally, we generate some steps with host and target information.
        //
        // The last step is by far the most complicated and subtle. The basic
        // thinking here is that we want to take the cartesian product of
        // specified hosts and targets and build rules with that. The list of
        // hosts and targets, if not specified, come from the how this build was
        // configured. If the rule we're looking at is a host-only rule the we
        // ignore the list of targets and instead consider the list of hosts
        // also the list of targets.
        //
        // Once the host and target lists are generated we take the cartesian
        // product of the two and then create a step based off them. Note that
        // the stage each step is associated was specified with the `--step`
        // flag on the command line.
        let (kind, paths) = match self.build.flags.cmd {
            Subcommand::Build { ref paths } => (Kind::Build, &paths[..]),
            Subcommand::Doc { ref paths } => (Kind::Doc, &paths[..]),
            Subcommand::Test { ref paths, test_args: _ } => (Kind::Test, &paths[..]),
            Subcommand::Bench { ref paths, test_args: _ } => (Kind::Bench, &paths[..]),
            Subcommand::Dist { ref paths, install } => {
                if install {
                    return vec![self.sbuild.name("install")]
                } else {
                    (Kind::Dist, &paths[..])
                }
            }
            Subcommand::Clean => panic!(),
        };

        self.rules.values().filter(|rule| rule.kind == kind).filter(|rule| {
            (paths.len() == 0 && rule.default) || paths.iter().any(|path| {
                path.ends_with(rule.path)
            })
        }).flat_map(|rule| {
            let hosts = if rule.only_host_build || rule.only_build {
                &self.build.config.host[..1]
            } else if self.build.flags.host.len() > 0 {
                &self.build.flags.host
            } else {
                &self.build.config.host
            };
            let targets = if self.build.flags.target.len() > 0 {
                &self.build.flags.target
            } else {
                &self.build.config.target
            };
            // Determine the actual targets participating in this rule.
            // NOTE: We should keep the full projection from build triple to
            // the hosts for the dist steps, now that the hosts array above is
            // truncated to avoid duplication of work in that case. Therefore
            // the original non-shadowed hosts array is used below.
            let arr = if rule.host {
                // If --target was specified but --host wasn't specified,
                // don't run any host-only tests. Also, respect any `--host`
                // overrides as done for `hosts`.
                if self.build.flags.host.len() > 0 {
                    &self.build.flags.host[..]
                } else if self.build.flags.target.len() > 0 {
                    &[]
                } else if rule.only_build {
                    &self.build.config.host[..1]
                } else {
                    &self.build.config.host[..]
                }
            } else {
                targets
            };

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
        let order = self.expand(steps);

        // Print out what we're doing for debugging
        self.build.verbose("bootstrap build plan:");
        for step in order.iter() {
            self.build.verbose(&format!("\t{:?}", step));
        }

        // And finally, iterate over everything and execute it.
        for step in order.iter() {
            if self.build.flags.keep_stage.map_or(false, |s| step.stage <= s) {
                self.build.verbose(&format!("keeping step {:?}", step));
                continue;
            }
            self.build.verbose(&format!("executing step {:?}", step));
            (self.rules[step.name].run)(step);
        }
    }

    /// From the top level targets `steps` generate a topological ordering of
    /// all steps needed to run those steps.
    fn expand(&self, steps: &[Step<'a>]) -> Vec<Step<'a>> {
        let mut order = Vec::new();
        let mut added = HashSet::new();
        added.insert(Step::noop());
        for step in steps.iter().cloned() {
            self.fill(step, &mut order, &mut added);
        }
        return order
    }

    /// Performs topological sort of dependencies rooted at the `step`
    /// specified, pushing all results onto the `order` vector provided.
    ///
    /// In other words, when this method returns, the `order` vector will
    /// contain a list of steps which if executed in order will eventually
    /// complete the `step` specified as well.
    ///
    /// The `added` set specified here is the set of steps that are already
    /// present in `order` (and hence don't need to be added again).
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
                let host = self.build.config.host.iter().any(|h| h == dep.target);
                let rules = self.rules.values().filter(|r| r.default);
                for rule in rules.filter(|r| r.kind == kind && (!r.host || host)) {
                    self.fill(dep.name(rule.name), order, added);
                }
            } else {
                self.fill(dep, order, added);
            }
        }
        order.push(step);
    }
}

#[cfg(test)]
mod tests {
    use std::env;

    use Build;
    use config::Config;
    use flags::Flags;

    macro_rules! a {
        ($($a:expr),*) => (vec![$($a.to_string()),*])
    }

    fn build(args: &[&str],
             extra_host: &[&str],
             extra_target: &[&str]) -> Build {
        let mut args = args.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        args.push("--build".to_string());
        args.push("A".to_string());
        let flags = Flags::parse(&args);

        let mut config = Config::default();
        config.docs = true;
        config.build = "A".to_string();
        config.host = vec![config.build.clone()];
        config.host.extend(extra_host.iter().map(|s| s.to_string()));
        config.target = config.host.clone();
        config.target.extend(extra_target.iter().map(|s| s.to_string()));

        let mut build = Build::new(flags, config);
        let cwd = env::current_dir().unwrap();
        build.crates.insert("std_shim".to_string(), ::Crate {
            name: "std_shim".to_string(),
            deps: Vec::new(),
            path: cwd.join("src/std_shim"),
            doc_step: "doc-std_shim".to_string(),
            build_step: "build-crate-std_shim".to_string(),
            test_step: "test-std_shim".to_string(),
            bench_step: "bench-std_shim".to_string(),
        });
        build.crates.insert("test_shim".to_string(), ::Crate {
            name: "test_shim".to_string(),
            deps: Vec::new(),
            path: cwd.join("src/test_shim"),
            doc_step: "doc-test_shim".to_string(),
            build_step: "build-crate-test_shim".to_string(),
            test_step: "test-test_shim".to_string(),
            bench_step: "bench-test_shim".to_string(),
        });
        build.crates.insert("rustc-main".to_string(), ::Crate {
            name: "rustc-main".to_string(),
            deps: Vec::new(),
            path: cwd.join("src/rustc-main"),
            doc_step: "doc-rustc-main".to_string(),
            build_step: "build-crate-rustc-main".to_string(),
            test_step: "test-rustc-main".to_string(),
            bench_step: "bench-rustc-main".to_string(),
        });
        return build
    }

    #[test]
    fn dist_baseline() {
        let build = build(&["dist"], &[], &[]);
        let rules = super::build_rules(&build);
        let plan = rules.plan();
        println!("rules: {:#?}", plan);
        assert!(plan.iter().all(|s| s.stage == 2));
        assert!(plan.iter().all(|s| s.host == "A" ));
        assert!(plan.iter().all(|s| s.target == "A" ));

        let step = super::Step {
            name: "",
            stage: 2,
            host: &build.config.build,
            target: &build.config.build,
        };

        assert!(plan.contains(&step.name("dist-docs")));
        assert!(plan.contains(&step.name("dist-mingw")));
        assert!(plan.contains(&step.name("dist-rustc")));
        assert!(plan.contains(&step.name("dist-std")));
        assert!(plan.contains(&step.name("dist-src")));
    }

    #[test]
    fn dist_with_targets() {
        let build = build(&["dist"], &[], &["B"]);
        let rules = super::build_rules(&build);
        let plan = rules.plan();
        println!("rules: {:#?}", plan);
        assert!(plan.iter().all(|s| s.stage == 2));
        assert!(plan.iter().all(|s| s.host == "A" ));

        let step = super::Step {
            name: "",
            stage: 2,
            host: &build.config.build,
            target: &build.config.build,
        };

        assert!(plan.contains(&step.name("dist-docs")));
        assert!(plan.contains(&step.name("dist-mingw")));
        assert!(plan.contains(&step.name("dist-rustc")));
        assert!(plan.contains(&step.name("dist-std")));
        assert!(plan.contains(&step.name("dist-src")));

        assert!(plan.contains(&step.target("B").name("dist-docs")));
        assert!(plan.contains(&step.target("B").name("dist-mingw")));
        assert!(!plan.contains(&step.target("B").name("dist-rustc")));
        assert!(plan.contains(&step.target("B").name("dist-std")));
        assert!(!plan.contains(&step.target("B").name("dist-src")));
    }

    #[test]
    fn dist_with_hosts() {
        let build = build(&["dist"], &["B"], &[]);
        let rules = super::build_rules(&build);
        let plan = rules.plan();
        println!("rules: {:#?}", plan);
        assert!(plan.iter().all(|s| s.stage == 2));

        let step = super::Step {
            name: "",
            stage: 2,
            host: &build.config.build,
            target: &build.config.build,
        };

        assert!(!plan.iter().any(|s| s.host == "B"));

        assert!(plan.contains(&step.name("dist-docs")));
        assert!(plan.contains(&step.name("dist-mingw")));
        assert!(plan.contains(&step.name("dist-rustc")));
        assert!(plan.contains(&step.name("dist-std")));
        assert!(plan.contains(&step.name("dist-src")));

        assert!(plan.contains(&step.target("B").name("dist-docs")));
        assert!(plan.contains(&step.target("B").name("dist-mingw")));
        assert!(plan.contains(&step.target("B").name("dist-rustc")));
        assert!(plan.contains(&step.target("B").name("dist-std")));
        assert!(!plan.contains(&step.target("B").name("dist-src")));
    }

    #[test]
    fn dist_with_targets_and_hosts() {
        let build = build(&["dist"], &["B"], &["C"]);
        let rules = super::build_rules(&build);
        let plan = rules.plan();
        println!("rules: {:#?}", plan);
        assert!(plan.iter().all(|s| s.stage == 2));

        let step = super::Step {
            name: "",
            stage: 2,
            host: &build.config.build,
            target: &build.config.build,
        };

        assert!(!plan.iter().any(|s| s.host == "B"));
        assert!(!plan.iter().any(|s| s.host == "C"));

        assert!(plan.contains(&step.name("dist-docs")));
        assert!(plan.contains(&step.name("dist-mingw")));
        assert!(plan.contains(&step.name("dist-rustc")));
        assert!(plan.contains(&step.name("dist-std")));
        assert!(plan.contains(&step.name("dist-src")));

        assert!(plan.contains(&step.target("B").name("dist-docs")));
        assert!(plan.contains(&step.target("B").name("dist-mingw")));
        assert!(plan.contains(&step.target("B").name("dist-rustc")));
        assert!(plan.contains(&step.target("B").name("dist-std")));
        assert!(!plan.contains(&step.target("B").name("dist-src")));

        assert!(plan.contains(&step.target("C").name("dist-docs")));
        assert!(plan.contains(&step.target("C").name("dist-mingw")));
        assert!(!plan.contains(&step.target("C").name("dist-rustc")));
        assert!(plan.contains(&step.target("C").name("dist-std")));
        assert!(!plan.contains(&step.target("C").name("dist-src")));
    }

    #[test]
    fn dist_target_with_target_flag() {
        let build = build(&["dist", "--target=C"], &["B"], &["C"]);
        let rules = super::build_rules(&build);
        let plan = rules.plan();
        println!("rules: {:#?}", plan);
        assert!(plan.iter().all(|s| s.stage == 2));

        let step = super::Step {
            name: "",
            stage: 2,
            host: &build.config.build,
            target: &build.config.build,
        };

        assert!(!plan.iter().any(|s| s.target == "A"));
        assert!(!plan.iter().any(|s| s.target == "B"));
        assert!(!plan.iter().any(|s| s.host == "B"));
        assert!(!plan.iter().any(|s| s.host == "C"));

        assert!(plan.contains(&step.target("C").name("dist-docs")));
        assert!(plan.contains(&step.target("C").name("dist-mingw")));
        assert!(!plan.contains(&step.target("C").name("dist-rustc")));
        assert!(plan.contains(&step.target("C").name("dist-std")));
        assert!(!plan.contains(&step.target("C").name("dist-src")));
    }

    #[test]
    fn dist_host_with_target_flag() {
        let build = build(&["dist", "--host=B", "--target=B"], &["B"], &["C"]);
        let rules = super::build_rules(&build);
        let plan = rules.plan();
        println!("rules: {:#?}", plan);
        assert!(plan.iter().all(|s| s.stage == 2));

        let step = super::Step {
            name: "",
            stage: 2,
            host: &build.config.build,
            target: &build.config.build,
        };

        assert!(!plan.iter().any(|s| s.target == "A"));
        assert!(!plan.iter().any(|s| s.target == "C"));
        assert!(!plan.iter().any(|s| s.host == "B"));
        assert!(!plan.iter().any(|s| s.host == "C"));

        assert!(plan.contains(&step.target("B").name("dist-docs")));
        assert!(plan.contains(&step.target("B").name("dist-mingw")));
        assert!(plan.contains(&step.target("B").name("dist-rustc")));
        assert!(plan.contains(&step.target("B").name("dist-std")));
        assert!(plan.contains(&step.target("B").name("dist-src")));

        let all = rules.expand(&plan);
        println!("all rules: {:#?}", all);
        assert!(!all.contains(&step.name("rustc")));
        assert!(!all.contains(&step.name("build-crate-std_shim").stage(1)));

        // all stage0 compiles should be for the build target, A
        for step in all.iter().filter(|s| s.stage == 0) {
            if !step.name.contains("build-crate") {
                continue
            }
            println!("step: {:?}", step);
            assert!(step.host != "B");
            assert!(step.target != "B");
            assert!(step.host != "C");
            assert!(step.target != "C");
        }
    }

    #[test]
    fn build_default() {
        let build = build(&["build"], &["B"], &["C"]);
        let rules = super::build_rules(&build);
        let plan = rules.plan();
        println!("rules: {:#?}", plan);
        assert!(plan.iter().all(|s| s.stage == 2));

        let step = super::Step {
            name: "",
            stage: 2,
            host: &build.config.build,
            target: &build.config.build,
        };

        // rustc built for all for of (A, B) x (A, B)
        assert!(plan.contains(&step.name("librustc")));
        assert!(plan.contains(&step.target("B").name("librustc")));
        assert!(plan.contains(&step.host("B").target("A").name("librustc")));
        assert!(plan.contains(&step.host("B").target("B").name("librustc")));

        // rustc never built for C
        assert!(!plan.iter().any(|s| {
            s.name.contains("rustc") && (s.host == "C" || s.target == "C")
        }));

        // test built for everything
        assert!(plan.contains(&step.name("libtest")));
        assert!(plan.contains(&step.target("B").name("libtest")));
        assert!(plan.contains(&step.host("B").target("A").name("libtest")));
        assert!(plan.contains(&step.host("B").target("B").name("libtest")));
        assert!(plan.contains(&step.host("A").target("C").name("libtest")));
        assert!(plan.contains(&step.host("B").target("C").name("libtest")));

        let all = rules.expand(&plan);
        println!("all rules: {:#?}", all);
        assert!(all.contains(&step.name("rustc")));
        assert!(all.contains(&step.name("libstd")));
    }

    #[test]
    fn build_filtered() {
        let build = build(&["build", "--target=C"], &["B"], &["C"]);
        let rules = super::build_rules(&build);
        let plan = rules.plan();
        println!("rules: {:#?}", plan);
        assert!(plan.iter().all(|s| s.stage == 2));

        assert!(!plan.iter().any(|s| s.name.contains("rustc")));
        assert!(plan.iter().all(|s| {
            !s.name.contains("test_shim") || s.target == "C"
        }));
    }

    #[test]
    fn test_default() {
        let build = build(&["test"], &[], &[]);
        let rules = super::build_rules(&build);
        let plan = rules.plan();
        println!("rules: {:#?}", plan);
        assert!(plan.iter().all(|s| s.stage == 2));
        assert!(plan.iter().all(|s| s.host == "A"));
        assert!(plan.iter().all(|s| s.target == "A"));

        assert!(plan.iter().any(|s| s.name.contains("-ui")));
        assert!(plan.iter().any(|s| s.name.contains("cfail")));
        assert!(plan.iter().any(|s| s.name.contains("cfail-full")));
        assert!(plan.iter().any(|s| s.name.contains("codegen-units")));
        assert!(plan.iter().any(|s| s.name.contains("debuginfo")));
        assert!(plan.iter().any(|s| s.name.contains("docs")));
        assert!(plan.iter().any(|s| s.name.contains("error-index")));
        assert!(plan.iter().any(|s| s.name.contains("incremental")));
        assert!(plan.iter().any(|s| s.name.contains("linkchecker")));
        assert!(plan.iter().any(|s| s.name.contains("mir-opt")));
        assert!(plan.iter().any(|s| s.name.contains("pfail")));
        assert!(plan.iter().any(|s| s.name.contains("rfail")));
        assert!(plan.iter().any(|s| s.name.contains("rfail-full")));
        assert!(plan.iter().any(|s| s.name.contains("rmake")));
        assert!(plan.iter().any(|s| s.name.contains("rpass")));
        assert!(plan.iter().any(|s| s.name.contains("rpass-full")));
        assert!(plan.iter().any(|s| s.name.contains("rustc-all")));
        assert!(plan.iter().any(|s| s.name.contains("rustdoc")));
        assert!(plan.iter().any(|s| s.name.contains("std-all")));
        assert!(plan.iter().any(|s| s.name.contains("test-all")));
        assert!(plan.iter().any(|s| s.name.contains("tidy")));
        assert!(plan.iter().any(|s| s.name.contains("valgrind")));
    }

    #[test]
    fn test_with_a_target() {
        let build = build(&["test", "--target=C"], &[], &["C"]);
        let rules = super::build_rules(&build);
        let plan = rules.plan();
        println!("rules: {:#?}", plan);
        assert!(plan.iter().all(|s| s.stage == 2));
        assert!(plan.iter().all(|s| s.host == "A"));
        assert!(plan.iter().all(|s| s.target == "C"));

        assert!(!plan.iter().any(|s| s.name.contains("-ui")));
        assert!(plan.iter().any(|s| s.name.contains("cfail")));
        assert!(!plan.iter().any(|s| s.name.contains("cfail-full")));
        assert!(plan.iter().any(|s| s.name.contains("codegen-units")));
        assert!(plan.iter().any(|s| s.name.contains("debuginfo")));
        assert!(!plan.iter().any(|s| s.name.contains("docs")));
        assert!(!plan.iter().any(|s| s.name.contains("error-index")));
        assert!(plan.iter().any(|s| s.name.contains("incremental")));
        assert!(!plan.iter().any(|s| s.name.contains("linkchecker")));
        assert!(plan.iter().any(|s| s.name.contains("mir-opt")));
        assert!(plan.iter().any(|s| s.name.contains("pfail")));
        assert!(plan.iter().any(|s| s.name.contains("rfail")));
        assert!(!plan.iter().any(|s| s.name.contains("rfail-full")));
        assert!(!plan.iter().any(|s| s.name.contains("rmake")));
        assert!(plan.iter().any(|s| s.name.contains("rpass")));
        assert!(!plan.iter().any(|s| s.name.contains("rpass-full")));
        assert!(!plan.iter().any(|s| s.name.contains("rustc-all")));
        assert!(!plan.iter().any(|s| s.name.contains("rustdoc")));
        assert!(plan.iter().any(|s| s.name.contains("std-all")));
        assert!(plan.iter().any(|s| s.name.contains("test-all")));
        assert!(!plan.iter().any(|s| s.name.contains("tidy")));
        assert!(plan.iter().any(|s| s.name.contains("valgrind")));
    }
}
