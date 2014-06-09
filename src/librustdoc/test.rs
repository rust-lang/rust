// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
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
use std::io::{Command, TempDir};
use std::os;
use std::str;
use std::string::String;
use std::dynamic_lib::DynamicLibrary;

use std::collections::{HashSet, HashMap};
use testing;
use rustc::back::link;
use rustc::driver::config;
use rustc::driver::driver;
use rustc::driver::session;
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

pub fn run(input: &str,
           cfgs: Vec<String>,
           libs: HashSet<Path>,
           mut test_args: Vec<String>)
           -> int {
    let input_path = Path::new(input);
    let input = driver::FileInput(input_path.clone());

    let sessopts = config::Options {
        maybe_sysroot: Some(os::self_exe_path().unwrap().dir_path()),
        addl_lib_search_paths: RefCell::new(libs.clone()),
        crate_types: vec!(config::CrateTypeDylib),
        ..config::basic_options().clone()
    };


    let codemap = CodeMap::new();
    let diagnostic_handler = diagnostic::default_handler(diagnostic::Auto);
    let span_diagnostic_handler =
    diagnostic::mk_span_handler(diagnostic_handler, codemap);

    let sess = session::build_session_(sessopts,
                                      Some(input_path.clone()),
                                      span_diagnostic_handler);

    let mut cfg = config::build_configuration(&sess);
    cfg.extend(cfgs.move_iter().map(|cfg_| {
        let cfg_ = token::intern_and_get_ident(cfg_.as_slice());
        @dummy_spanned(ast::MetaWord(cfg_))
    }));
    let krate = driver::phase_1_parse_input(&sess, cfg, &input);
    let (krate, _) = driver::phase_2_configure_and_expand(&sess, krate,
                                                          &from_str("rustdoc-test").unwrap());

    let ctx = @core::DocContext {
        krate: krate,
        maybe_typed: core::NotTyped(sess),
        src: input_path,
        external_paths: RefCell::new(Some(HashMap::new())),
        external_traits: RefCell::new(None),
        external_typarams: RefCell::new(None),
        inlined: RefCell::new(None),
        populated_crate_impls: RefCell::new(HashSet::new()),
    };
    super::ctxtkey.replace(Some(ctx));

    let mut v = RustdocVisitor::new(ctx, None);
    v.visit(&ctx.krate);
    let krate = v.clean();
    let (krate, _) = passes::unindent_comments(krate);
    let (krate, _) = passes::collapse_docs(krate);

    let mut collector = Collector::new(krate.name.to_string(),
                                       libs,
                                       false);
    collector.fold_crate(krate);

    test_args.unshift("rustdoctest".to_string());

    testing::test_main(test_args.as_slice(),
                       collector.tests.move_iter().collect());
    0
}

fn runtest(test: &str, cratename: &str, libs: HashSet<Path>, should_fail: bool,
           no_run: bool) {
    let test = maketest(test, Some(cratename), true);
    let input = driver::StrInput(test.to_string());

    let sessopts = config::Options {
        maybe_sysroot: Some(os::self_exe_path().unwrap().dir_path()),
        addl_lib_search_paths: RefCell::new(libs),
        crate_types: vec!(config::CrateTypeExecutable),
        output_types: vec!(link::OutputTypeExe),
        no_trans: no_run,
        cg: config::CodegenOptions {
            prefer_dynamic: true,
            .. config::basic_codegen_options()
        },
        ..config::basic_options().clone()
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
    let old = io::stdio::set_stderr(box w1);
    spawn(proc() {
        let mut p = io::ChanReader::new(rx);
        let mut err = old.unwrap_or(box io::stderr() as Box<Writer:Send>);
        io::util::copy(&mut p, &mut err).unwrap();
    });
    let emitter = diagnostic::EmitterWriter::new(box w2);

    // Compile the code
    let codemap = CodeMap::new();
    let diagnostic_handler = diagnostic::mk_handler(box emitter);
    let span_diagnostic_handler =
        diagnostic::mk_span_handler(diagnostic_handler, codemap);

    let sess = session::build_session_(sessopts,
                                      None,
                                      span_diagnostic_handler);

    let outdir = TempDir::new("rustdoctest").expect("rustdoc needs a tempdir");
    let out = Some(outdir.path().clone());
    let cfg = config::build_configuration(&sess);
    let libdir = sess.target_filesearch().get_lib_path();
    driver::compile_input(sess, cfg, &input, &out, &None);

    if no_run { return }

    // Run the code!
    //
    // We're careful to prepend the *target* dylib search path to the child's
    // environment to ensure that the target loads the right libraries at
    // runtime. It would be a sad day if the *host* libraries were loaded as a
    // mistake.
    let exe = outdir.path().join("rust_out");
    let env = {
        let mut path = DynamicLibrary::search_path();
        path.insert(0, libdir.clone());

        // Remove the previous dylib search path var
        let var = DynamicLibrary::envvar();
        let mut env: Vec<(String,String)> = os::env().move_iter().collect();
        match env.iter().position(|&(ref k, _)| k.as_slice() == var) {
            Some(i) => { env.remove(i); }
            None => {}
        };

        // Add the new dylib search path var
        let newpath = DynamicLibrary::create_path(path.as_slice());
        env.push((var.to_string(),
                  str::from_utf8(newpath.as_slice()).unwrap().to_string()));
        env
    };
    match Command::new(exe).env(env.as_slice()).output() {
        Err(e) => fail!("couldn't run the test: {}{}", e,
                        if e.kind == io::PermissionDenied {
                            " - maybe your tempdir is mounted with noexec?"
                        } else { "" }),
        Ok(out) => {
            if should_fail && out.status.success() {
                fail!("test executable succeeded when it should have failed");
            } else if !should_fail && !out.status.success() {
                fail!("test executable failed:\n{}",
                      str::from_utf8(out.error.as_slice()));
            }
        }
    }
}

pub fn maketest(s: &str, cratename: Option<&str>, lints: bool) -> String {
    let mut prog = String::new();
    if lints {
        prog.push_str(r"
#![deny(warnings)]
#![allow(unused_variable, dead_assignment, unused_mut, unused_attribute, dead_code)]
");
    }

    if !s.contains("extern crate") {
        match cratename {
            Some(cratename) => {
                if s.contains(cratename) {
                    prog.push_str(format!("extern crate {};\n",
                                          cratename).as_slice());
                }
            }
            None => {}
        }
    }
    if s.contains("fn main") {
        prog.push_str(s);
    } else {
        prog.push_str("fn main() {\n    ");
        prog.push_str(s.replace("\n", "\n    ").as_slice());
        prog.push_str("\n}");
    }

    return prog
}

pub struct Collector {
    pub tests: Vec<testing::TestDescAndFn>,
    names: Vec<String>,
    libs: HashSet<Path>,
    cnt: uint,
    use_headers: bool,
    current_header: Option<String>,
    cratename: String,
}

impl Collector {
    pub fn new(cratename: String, libs: HashSet<Path>,
               use_headers: bool) -> Collector {
        Collector {
            tests: Vec::new(),
            names: Vec::new(),
            libs: libs,
            cnt: 0,
            use_headers: use_headers,
            current_header: None,
            cratename: cratename,
        }
    }

    pub fn add_test(&mut self, test: String, should_fail: bool, no_run: bool, should_ignore: bool) {
        let name = if self.use_headers {
            let s = self.current_header.as_ref().map(|s| s.as_slice()).unwrap_or("");
            format!("{}_{}", s, self.cnt)
        } else {
            format!("{}_{}", self.names.connect("::"), self.cnt)
        };
        self.cnt += 1;
        let libs = self.libs.clone();
        let cratename = self.cratename.to_string();
        debug!("Creating test {}: {}", name, test);
        self.tests.push(testing::TestDescAndFn {
            desc: testing::TestDesc {
                name: testing::DynTestName(name),
                ignore: should_ignore,
                should_fail: false, // compiler failures are test failures
            },
            testfn: testing::DynTestFn(proc() {
                runtest(test.as_slice(),
                        cratename.as_slice(),
                        libs,
                        should_fail,
                        no_run);
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
                }).collect::<String>();

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
            Some(ref name) => { self.names.push(name.to_string()); true }
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
