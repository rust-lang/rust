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
use std::char;
use std::io;
use std::io::{Process, TempDir};
use std::local_data;
use std::os;
use std::str;

use collections::HashSet;
use testing;
use rustc::back::link;
use rustc::driver::driver;
use rustc::driver::session;
use rustc::metadata::creader::Loader;
use syntax::ast;
use syntax::codemap::{CodeMap, dummy_spanned};
use syntax::diagnostic;
use syntax::parse::token;

use core;
use clean;
use clean::Clean;
use fold::DocFolder;
use html::markdown;
use passes;
use visit_ast::RustdocVisitor;

pub fn run(input: &str, cfgs: Vec<~str>,
           libs: HashSet<Path>, mut test_args: Vec<~str>) -> int {
    let input_path = Path::new(input);
    let input = driver::FileInput(input_path.clone());

    let sessopts = session::Options {
        maybe_sysroot: Some(os::self_exe_path().unwrap().dir_path()),
        addl_lib_search_paths: RefCell::new(libs.clone()),
        crate_types: vec!(session::CrateTypeDylib),
        ..session::basic_options().clone()
    };


    let codemap = CodeMap::new();
    let diagnostic_handler = diagnostic::default_handler();
    let span_diagnostic_handler =
    diagnostic::mk_span_handler(diagnostic_handler, codemap);

    let sess = driver::build_session_(sessopts,
                                      Some(input_path),
                                      span_diagnostic_handler);

    let mut cfg = driver::build_configuration(&sess);
    cfg.extend(cfgs.move_iter().map(|cfg_| {
        let cfg_ = token::intern_and_get_ident(cfg_);
        @dummy_spanned(ast::MetaWord(cfg_))
    }));
    let krate = driver::phase_1_parse_input(&sess, cfg, &input);
    let (krate, _) = driver::phase_2_configure_and_expand(&sess, &mut Loader::new(&sess), krate,
                                                          &from_str("rustdoc-test").unwrap());

    let ctx = @core::DocContext {
        krate: krate,
        maybe_typed: core::NotTyped(sess),
    };
    local_data::set(super::ctxtkey, ctx);

    let mut v = RustdocVisitor::new(ctx, None);
    v.visit(&ctx.krate);
    let krate = v.clean();
    let (krate, _) = passes::unindent_comments(krate);
    let (krate, _) = passes::collapse_docs(krate);

    let mut collector = Collector::new(krate.name.to_owned(),
                                       libs,
                                       false,
                                       false);
    collector.fold_crate(krate);

    test_args.unshift(~"rustdoctest");

    testing::test_main(test_args.as_slice(),
                       collector.tests.move_iter().collect());
    0
}

fn runtest(test: &str, cratename: &str, libs: HashSet<Path>, should_fail: bool,
           no_run: bool, loose_feature_gating: bool) {
    let test = maketest(test, cratename, loose_feature_gating);
    let input = driver::StrInput(test);

    let sessopts = session::Options {
        maybe_sysroot: Some(os::self_exe_path().unwrap().dir_path()),
        addl_lib_search_paths: RefCell::new(libs),
        crate_types: vec!(session::CrateTypeExecutable),
        output_types: vec!(link::OutputTypeExe),
        cg: session::CodegenOptions {
            prefer_dynamic: true,
            .. session::basic_codegen_options()
        },
        ..session::basic_options().clone()
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
    let (tx, rx) = channel();
    let w1 = io::ChanWriter::new(tx);
    let w2 = w1.clone();
    let old = io::stdio::set_stderr(~w1);
    spawn(proc() {
        let mut p = io::ChanReader::new(rx);
        let mut err = old.unwrap_or(~io::stderr() as ~Writer:Send);
        io::util::copy(&mut p, &mut err).unwrap();
    });
    let emitter = diagnostic::EmitterWriter::new(~w2);

    // Compile the code
    let codemap = CodeMap::new();
    let diagnostic_handler = diagnostic::mk_handler(~emitter);
    let span_diagnostic_handler =
        diagnostic::mk_span_handler(diagnostic_handler, codemap);

    let sess = driver::build_session_(sessopts,
                                      None,
                                      span_diagnostic_handler);

    let outdir = TempDir::new("rustdoctest").expect("rustdoc needs a tempdir");
    let out = Some(outdir.path().clone());
    let cfg = driver::build_configuration(&sess);
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

fn maketest(s: &str, cratename: &str, loose_feature_gating: bool) -> ~str {
    let mut prog = ~r"
#![deny(warnings)]
#![allow(unused_variable, dead_assignment, unused_mut, attribute_usage, dead_code)]
";

    if loose_feature_gating {
        // FIXME #12773: avoid inserting these when the tutorial & manual
        // etc. have been updated to not use them so prolifically.
        prog.push_str("#![feature(macro_rules, globs, struct_variant, managed_boxes) ]\n");
    }

    if !s.contains("extern crate") {
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
    pub tests: Vec<testing::TestDescAndFn>,
    names: Vec<~str>,
    libs: HashSet<Path>,
    cnt: uint,
    use_headers: bool,
    current_header: Option<~str>,
    cratename: ~str,

    loose_feature_gating: bool
}

impl Collector {
    pub fn new(cratename: ~str, libs: HashSet<Path>,
               use_headers: bool, loose_feature_gating: bool) -> Collector {
        Collector {
            tests: Vec::new(),
            names: Vec::new(),
            libs: libs,
            cnt: 0,
            use_headers: use_headers,
            current_header: None,
            cratename: cratename,

            loose_feature_gating: loose_feature_gating
        }
    }

    pub fn add_test(&mut self, test: ~str, should_fail: bool, no_run: bool) {
        let name = if self.use_headers {
            let s = self.current_header.as_ref().map(|s| s.as_slice()).unwrap_or("");
            format!("{}_{}", s, self.cnt)
        } else {
            format!("{}_{}", self.names.connect("::"), self.cnt)
        };
        self.cnt += 1;
        let libs = self.libs.clone();
        let cratename = self.cratename.to_owned();
        let loose_feature_gating = self.loose_feature_gating;
        debug!("Creating test {}: {}", name, test);
        self.tests.push(testing::TestDescAndFn {
            desc: testing::TestDesc {
                name: testing::DynTestName(name),
                ignore: false,
                should_fail: false, // compiler failures are test failures
            },
            testfn: testing::DynTestFn(proc() {
                runtest(test, cratename, libs, should_fail, no_run, loose_feature_gating);
            }),
        });
    }

    pub fn register_header(&mut self, name: &str, level: u32) {
        if self.use_headers && level == 1 {
            // we use these headings as test names, so it's good if
            // they're valid identifiers.
            let name = name.chars().enumerate().map(|(i, c)| {
                    if (i == 0 && char::is_XID_start(c)) ||
                        (i != 0 && char::is_XID_continue(c)) {
                        c
                    } else {
                        '_'
                    }
                }).collect::<~str>();

            // new header => reset count.
            self.cnt = 0;
            self.current_header = Some(name);
        }
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
                markdown::find_testable_code(doc, &mut *self);
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
