// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::RefCell;
use std::io;
use std::io::Process;
use std::local_data;
use std::os;
use std::str;

use collections::HashSet;
use testing;
use extra::tempfile::TempDir;
use rustc::back::link;
use rustc::driver::driver;
use rustc::driver::session;
use rustc::metadata::creader::Loader;
use getopts;
use syntax::diagnostic;
use syntax::parse;
use syntax::codemap::CodeMap;

use core;
use clean;
use clean::Clean;
use fold::DocFolder;
use html::markdown;
use passes;
use visit_ast::RustdocVisitor;

pub fn run(input: &str, matches: &getopts::Matches) -> int {
    let input_path = Path::new(input);
    let input = driver::FileInput(input_path.clone());
    let libs = matches.opt_strs("L").map(|s| Path::new(s.as_slice()));
    let libs = @RefCell::new(libs.move_iter().collect());

    let sessopts = @session::Options {
        maybe_sysroot: Some(@os::self_exe_path().unwrap().dir_path()),
        addl_lib_search_paths: libs,
        crate_types: vec!(session::CrateTypeDylib),
        .. (*session::basic_options()).clone()
    };


    let cm = @CodeMap::new();
    let diagnostic_handler = diagnostic::default_handler();
    let span_diagnostic_handler =
        diagnostic::mk_span_handler(diagnostic_handler, cm);
    let parsesess = parse::new_parse_sess_special_handler(span_diagnostic_handler,
                                                          cm);

    let sess = driver::build_session_(sessopts,
                                      Some(input_path),
                                      parsesess.cm,
                                      span_diagnostic_handler);

    let cfg = driver::build_configuration(sess);
    let krate = driver::phase_1_parse_input(sess, cfg, &input);
    let loader = &mut Loader::new(sess);
    let (krate, _) = driver::phase_2_configure_and_expand(sess, loader, krate);

    let ctx = @core::DocContext {
        krate: krate,
        tycx: None,
        sess: sess,
    };
    local_data::set(super::ctxtkey, ctx);

    let mut v = RustdocVisitor::new(ctx, None);
    v.visit(&ctx.krate);
    let krate = v.clean();
    let (krate, _) = passes::unindent_comments(krate);
    let (krate, _) = passes::collapse_docs(krate);

    let mut collector = Collector {
        tests: ~[],
        names: ~[],
        cnt: 0,
        libs: libs,
        cratename: krate.name.to_owned(),
    };
    collector.fold_crate(krate);

    let args = matches.opt_strs("test-args");
    let mut args = args.iter().flat_map(|s| s.words()).map(|s| s.to_owned());
    let mut args = args.to_owned_vec();
    args.unshift(~"rustdoctest");

    testing::test_main(args, collector.tests);

    0
}

fn runtest(test: &str, cratename: &str, libs: HashSet<Path>, should_fail: bool,
           no_run: bool) {
    let test = maketest(test, cratename);
    let parsesess = parse::new_parse_sess();
    let input = driver::StrInput(test);

    let sessopts = @session::Options {
        maybe_sysroot: Some(@os::self_exe_path().unwrap().dir_path()),
        addl_lib_search_paths: @RefCell::new(libs),
        crate_types: vec!(session::CrateTypeExecutable),
        output_types: vec!(link::OutputTypeExe),
        cg: session::CodegenOptions {
            prefer_dynamic: true,
            .. session::basic_codegen_options()
        },
        .. (*session::basic_options()).clone()
    };

    // Shuffle around a few input and output handles here. We're going to pass
    // an explicit handle into rustc to collect output messages, but we also
    // want to catch the error message that rustc prints when it fails.
    //
    // We take our task-local stderr (likely set by the test runner), and move
    // it into another task. This helper task then acts as a sink for both the
    // stderr of this task and stderr of rustc itself, copying all the info onto
    // the stderr channel we originally started with.
    //
    // The basic idea is to not use a default_handler() for rustc, and then also
    // not print things by default to the actual stderr.
    let (p, c) = Chan::new();
    let w1 = io::ChanWriter::new(c);
    let w2 = w1.clone();
    let old = io::stdio::set_stderr(~w1);
    spawn(proc() {
        let mut p = io::PortReader::new(p);
        let mut err = old.unwrap_or(~io::stderr() as ~Writer);
        io::util::copy(&mut p, &mut err).unwrap();
    });
    let emitter = diagnostic::EmitterWriter::new(~w2);

    // Compile the code
    let diagnostic_handler = diagnostic::mk_handler(~emitter);
    let span_diagnostic_handler =
        diagnostic::mk_span_handler(diagnostic_handler, parsesess.cm);

    let sess = driver::build_session_(sessopts,
                                      None,
                                      parsesess.cm,
                                      span_diagnostic_handler);

    let outdir = TempDir::new("rustdoctest").expect("rustdoc needs a tempdir");
    let out = Some(outdir.path().clone());
    let cfg = driver::build_configuration(sess);
    driver::compile_input(sess, cfg, &input, &out, &None);

    if no_run { return }

    // Run the code!
    let exe = outdir.path().join("rust_out");
    let out = Process::output(exe.as_str().unwrap(), []);
    match out {
        Err(e) => fail!("couldn't run the test: {}{}", e,
                        if e.kind == io::PermissionDenied {
                            " - maybe your tempdir is mounted with noexec?"
                        } else { "" }),
        Ok(out) => {
            if should_fail && out.status.success() {
                fail!("test executable succeeded when it should have failed");
            } else if !should_fail && !out.status.success() {
                fail!("test executable failed:\n{}", str::from_utf8(out.error));
            }
        }
    }
}

fn maketest(s: &str, cratename: &str) -> ~str {
    let mut prog = ~r"
#[deny(warnings)];
#[allow(unused_variable, dead_assignment, unused_mut, attribute_usage, dead_code)];
";
    if !s.contains("extern crate") {
        if s.contains("extra") {
            prog.push_str("extern crate extra;\n");
        }
        if s.contains(cratename) {
            prog.push_str(format!("extern crate {};\n", cratename));
        }
    }
    if s.contains("fn main") {
        prog.push_str(s);
    } else {
        prog.push_str("fn main() {\n");
        prog.push_str(s);
        prog.push_str("\n}");
    }

    return prog;
}

pub struct Collector {
    priv tests: ~[testing::TestDescAndFn],
    priv names: ~[~str],
    priv libs: @RefCell<HashSet<Path>>,
    priv cnt: uint,
    priv cratename: ~str,
}

impl Collector {
    pub fn add_test(&mut self, test: &str, should_fail: bool, no_run: bool) {
        let test = test.to_owned();
        let name = format!("{}_{}", self.names.connect("::"), self.cnt);
        self.cnt += 1;
        let libs = self.libs.borrow();
        let libs = (*libs.get()).clone();
        let cratename = self.cratename.to_owned();
        debug!("Creating test {}: {}", name, test);
        self.tests.push(testing::TestDescAndFn {
            desc: testing::TestDesc {
                name: testing::DynTestName(name),
                ignore: false,
                should_fail: false, // compiler failures are test failures
            },
            testfn: testing::DynTestFn(proc() {
                runtest(test, cratename, libs, should_fail, no_run);
            }),
        });
    }
}

impl DocFolder for Collector {
    fn fold_item(&mut self, item: clean::Item) -> Option<clean::Item> {
        let pushed = match item.name {
            Some(ref name) if name.len() == 0 => false,
            Some(ref name) => { self.names.push(name.to_owned()); true }
            None => false
        };
        match item.doc_value() {
            Some(doc) => {
                self.cnt = 0;
                markdown::find_testable_code(doc, self);
            }
            None => {}
        }
        let ret = self.fold_item_recur(item);
        if pushed {
            self.names.pop();
        }
        return ret;
    }
}
