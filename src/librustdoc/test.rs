// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(deprecated)]

use std::cell::{RefCell, Cell};
use std::collections::{HashSet, HashMap};
use std::dynamic_lib::DynamicLibrary;
use std::env;
use std::ffi::OsString;
use std::io::prelude::*;
use std::io;
use std::path::PathBuf;
use std::process::Command;
use std::rc::Rc;
use std::str;
use std::sync::{Arc, Mutex};

use testing;
use rustc_lint;
use rustc::front::map as hir_map;
use rustc::session::{self, config};
use rustc::session::config::{get_unstable_features_setting, OutputType};
use rustc::session::search_paths::{SearchPaths, PathKind};
use rustc_front::lowering::{lower_crate, LoweringContext};
use rustc_back::tempdir::TempDir;
use rustc_driver::{driver, Compilation};
use rustc_metadata::cstore::CStore;
use syntax::codemap::CodeMap;
use syntax::errors;
use syntax::errors::emitter::ColorConfig;
use syntax::parse::token;

use core;
use clean;
use clean::Clean;
use fold::DocFolder;
use html::markdown;
use passes;
use visit_ast::RustdocVisitor;

#[derive(Clone, Default)]
pub struct TestOptions {
    pub no_crate_inject: bool,
    pub attrs: Vec<String>,
}

pub fn run(input: &str,
           cfgs: Vec<String>,
           libs: SearchPaths,
           externs: core::Externs,
           mut test_args: Vec<String>,
           crate_name: Option<String>)
           -> isize {
    let input_path = PathBuf::from(input);
    let input = config::Input::File(input_path.clone());

    let sessopts = config::Options {
        maybe_sysroot: Some(env::current_exe().unwrap().parent().unwrap()
                                              .parent().unwrap().to_path_buf()),
        search_paths: libs.clone(),
        crate_types: vec!(config::CrateTypeDylib),
        externs: externs.clone(),
        unstable_features: get_unstable_features_setting(),
        ..config::basic_options().clone()
    };

    let codemap = Rc::new(CodeMap::new());
    let diagnostic_handler = errors::Handler::new(ColorConfig::Auto,
                                                  None,
                                                  true,
                                                  false,
                                                  codemap.clone());

    let cstore = Rc::new(CStore::new(token::get_ident_interner()));
    let cstore_ = ::rustc_driver::cstore_to_cratestore(cstore.clone());
    let sess = session::build_session_(sessopts,
                                       Some(input_path.clone()),
                                       diagnostic_handler,
                                       codemap,
                                       cstore_);
    rustc_lint::register_builtins(&mut sess.lint_store.borrow_mut(), Some(&sess));

    let mut cfg = config::build_configuration(&sess);
    cfg.extend(config::parse_cfgspecs(cfgs.clone()));
    let krate = driver::phase_1_parse_input(&sess, cfg, &input);
    let krate = driver::phase_2_configure_and_expand(&sess, &cstore, krate,
                                                     "rustdoc-test", None)
        .expect("phase_2_configure_and_expand aborted in rustdoc!");
    let krate = driver::assign_node_ids(&sess, krate);
    let lcx = LoweringContext::new(&sess, Some(&krate));
    let krate = lower_crate(&lcx, &krate);

    let opts = scrape_test_config(&krate);

    let mut forest = hir_map::Forest::new(krate);
    let map = hir_map::map_crate(&mut forest);

    let ctx = core::DocContext {
        map: &map,
        maybe_typed: core::NotTyped(&sess),
        input: input,
        external_paths: RefCell::new(Some(HashMap::new())),
        external_traits: RefCell::new(None),
        external_typarams: RefCell::new(None),
        inlined: RefCell::new(None),
        populated_crate_impls: RefCell::new(HashSet::new()),
        deref_trait_did: Cell::new(None),
    };

    let mut v = RustdocVisitor::new(&ctx, None);
    v.visit(ctx.map.krate());
    let mut krate = v.clean(&ctx);
    match crate_name {
        Some(name) => krate.name = name,
        None => {}
    }
    let (krate, _) = passes::collapse_docs(krate);
    let (krate, _) = passes::unindent_comments(krate);

    let mut collector = Collector::new(krate.name.to_string(),
                                       cfgs,
                                       libs,
                                       externs,
                                       false,
                                       opts);
    collector.fold_crate(krate);

    test_args.insert(0, "rustdoctest".to_string());

    testing::test_main(&test_args,
                       collector.tests.into_iter().collect());
    0
}

// Look for #![doc(test(no_crate_inject))], used by crates in the std facade
fn scrape_test_config(krate: &::rustc_front::hir::Crate) -> TestOptions {
    use syntax::attr::AttrMetaMethods;
    use syntax::print::pprust;

    let mut opts = TestOptions {
        no_crate_inject: false,
        attrs: Vec::new(),
    };

    let attrs = krate.attrs.iter()
                     .filter(|a| a.check_name("doc"))
                     .filter_map(|a| a.meta_item_list())
                     .flat_map(|l| l)
                     .filter(|a| a.check_name("test"))
                     .filter_map(|a| a.meta_item_list())
                     .flat_map(|l| l);
    for attr in attrs {
        if attr.check_name("no_crate_inject") {
            opts.no_crate_inject = true;
        }
        if attr.check_name("attr") {
            if let Some(l) = attr.meta_item_list() {
                for item in l {
                    opts.attrs.push(pprust::meta_item_to_string(item));
                }
            }
        }
    }

    return opts;
}

fn runtest(test: &str, cratename: &str, cfgs: Vec<String>, libs: SearchPaths,
           externs: core::Externs,
           should_panic: bool, no_run: bool, as_test_harness: bool,
           opts: &TestOptions) {
    // the test harness wants its own `main` & top level functions, so
    // never wrap the test in `fn main() { ... }`
    let test = maketest(test, Some(cratename), as_test_harness, opts);
    let input = config::Input::Str(test.to_string());
    let mut outputs = HashMap::new();
    outputs.insert(OutputType::Exe, None);

    let sessopts = config::Options {
        maybe_sysroot: Some(env::current_exe().unwrap().parent().unwrap()
                                              .parent().unwrap().to_path_buf()),
        search_paths: libs,
        crate_types: vec!(config::CrateTypeExecutable),
        output_types: outputs,
        externs: externs,
        cg: config::CodegenOptions {
            prefer_dynamic: true,
            .. config::basic_codegen_options()
        },
        test: as_test_harness,
        unstable_features: get_unstable_features_setting(),
        ..config::basic_options().clone()
    };

    // Shuffle around a few input and output handles here. We're going to pass
    // an explicit handle into rustc to collect output messages, but we also
    // want to catch the error message that rustc prints when it fails.
    //
    // We take our thread-local stderr (likely set by the test runner) and replace
    // it with a sink that is also passed to rustc itself. When this function
    // returns the output of the sink is copied onto the output of our own thread.
    //
    // The basic idea is to not use a default Handler for rustc, and then also
    // not print things by default to the actual stderr.
    struct Sink(Arc<Mutex<Vec<u8>>>);
    impl Write for Sink {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            Write::write(&mut *self.0.lock().unwrap(), data)
        }
        fn flush(&mut self) -> io::Result<()> { Ok(()) }
    }
    struct Bomb(Arc<Mutex<Vec<u8>>>, Box<Write+Send>);
    impl Drop for Bomb {
        fn drop(&mut self) {
            let _ = self.1.write_all(&self.0.lock().unwrap());
        }
    }
    let data = Arc::new(Mutex::new(Vec::new()));
    let codemap = Rc::new(CodeMap::new());
    let emitter = errors::emitter::EmitterWriter::new(box Sink(data.clone()), None, codemap.clone());
    let old = io::set_panic(box Sink(data.clone()));
    let _bomb = Bomb(data, old.unwrap_or(box io::stdout()));

    // Compile the code
    let diagnostic_handler = errors::Handler::with_emitter(true, false, box emitter);

    let cstore = Rc::new(CStore::new(token::get_ident_interner()));
    let cstore_ = ::rustc_driver::cstore_to_cratestore(cstore.clone());
    let sess = session::build_session_(sessopts,
                                       None,
                                       diagnostic_handler,
                                       codemap,
                                       cstore_);
    rustc_lint::register_builtins(&mut sess.lint_store.borrow_mut(), Some(&sess));

    let outdir = TempDir::new("rustdoctest").ok().expect("rustdoc needs a tempdir");
    let out = Some(outdir.path().to_path_buf());
    let mut cfg = config::build_configuration(&sess);
    cfg.extend(config::parse_cfgspecs(cfgs));
    let libdir = sess.target_filesearch(PathKind::All).get_lib_path();
    let mut control = driver::CompileController::basic();
    if no_run {
        control.after_analysis.stop = Compilation::Stop;
    }
    driver::compile_input(sess, &cstore, cfg, &input, &out, &None, None, control);

    if no_run { return }

    // Run the code!
    //
    // We're careful to prepend the *target* dylib search path to the child's
    // environment to ensure that the target loads the right libraries at
    // runtime. It would be a sad day if the *host* libraries were loaded as a
    // mistake.
    let mut cmd = Command::new(&outdir.path().join("rust_out"));
    let var = DynamicLibrary::envvar();
    let newpath = {
        let path = env::var_os(var).unwrap_or(OsString::new());
        let mut path = env::split_paths(&path).collect::<Vec<_>>();
        path.insert(0, libdir.clone());
        env::join_paths(path).unwrap()
    };
    cmd.env(var, &newpath);

    match cmd.output() {
        Err(e) => panic!("couldn't run the test: {}{}", e,
                        if e.kind() == io::ErrorKind::PermissionDenied {
                            " - maybe your tempdir is mounted with noexec?"
                        } else { "" }),
        Ok(out) => {
            if should_panic && out.status.success() {
                panic!("test executable succeeded when it should have failed");
            } else if !should_panic && !out.status.success() {
                panic!("test executable failed:\n{}\n{}",
                       str::from_utf8(&out.stdout).unwrap_or(""),
                       str::from_utf8(&out.stderr).unwrap_or(""));
            }
        }
    }
}

pub fn maketest(s: &str, cratename: Option<&str>, dont_insert_main: bool,
                opts: &TestOptions) -> String {
    let (crate_attrs, everything_else) = partition_source(s);

    let mut prog = String::new();

    // First push any outer attributes from the example, assuming they
    // are intended to be crate attributes.
    prog.push_str(&crate_attrs);

    // Next, any attributes for other aspects such as lints.
    for attr in &opts.attrs {
        prog.push_str(&format!("#![{}]\n", attr));
    }

    // Don't inject `extern crate std` because it's already injected by the
    // compiler.
    if !s.contains("extern crate") && !opts.no_crate_inject && cratename != Some("std") {
        match cratename {
            Some(cratename) => {
                if s.contains(cratename) {
                    prog.push_str(&format!("extern crate {};\n", cratename));
                }
            }
            None => {}
        }
    }
    if dont_insert_main || s.contains("fn main") {
        prog.push_str(&everything_else);
    } else {
        prog.push_str("fn main() {\n    ");
        prog.push_str(&everything_else.replace("\n", "\n    "));
        prog.push_str("\n}");
    }

    info!("final test program: {}", prog);

    return prog
}

fn partition_source(s: &str) -> (String, String) {
    use rustc_unicode::str::UnicodeStr;

    let mut after_header = false;
    let mut before = String::new();
    let mut after = String::new();

    for line in s.lines() {
        let trimline = line.trim();
        let header = trimline.is_whitespace() ||
            trimline.starts_with("#![feature");
        if !header || after_header {
            after_header = true;
            after.push_str(line);
            after.push_str("\n");
        } else {
            before.push_str(line);
            before.push_str("\n");
        }
    }

    return (before, after);
}

pub struct Collector {
    pub tests: Vec<testing::TestDescAndFn>,
    names: Vec<String>,
    cfgs: Vec<String>,
    libs: SearchPaths,
    externs: core::Externs,
    cnt: usize,
    use_headers: bool,
    current_header: Option<String>,
    cratename: String,
    opts: TestOptions,
}

impl Collector {
    pub fn new(cratename: String, cfgs: Vec<String>, libs: SearchPaths, externs: core::Externs,
               use_headers: bool, opts: TestOptions) -> Collector {
        Collector {
            tests: Vec::new(),
            names: Vec::new(),
            cfgs: cfgs,
            libs: libs,
            externs: externs,
            cnt: 0,
            use_headers: use_headers,
            current_header: None,
            cratename: cratename,
            opts: opts,
        }
    }

    pub fn add_test(&mut self, test: String,
                    should_panic: bool, no_run: bool, should_ignore: bool,
                    as_test_harness: bool) {
        let name = if self.use_headers {
            let s = self.current_header.as_ref().map(|s| &**s).unwrap_or("");
            format!("{}_{}", s, self.cnt)
        } else {
            format!("{}_{}", self.names.join("::"), self.cnt)
        };
        self.cnt += 1;
        let cfgs = self.cfgs.clone();
        let libs = self.libs.clone();
        let externs = self.externs.clone();
        let cratename = self.cratename.to_string();
        let opts = self.opts.clone();
        debug!("Creating test {}: {}", name, test);
        self.tests.push(testing::TestDescAndFn {
            desc: testing::TestDesc {
                name: testing::DynTestName(name),
                ignore: should_ignore,
                // compiler failures are test failures
                should_panic: testing::ShouldPanic::No,
            },
            testfn: testing::DynTestFn(Box::new(move|| {
                runtest(&test,
                        &cratename,
                        cfgs,
                        libs,
                        externs,
                        should_panic,
                        no_run,
                        as_test_harness,
                        &opts);
            }))
        });
    }

    pub fn register_header(&mut self, name: &str, level: u32) {
        if self.use_headers && level == 1 {
            // we use these headings as test names, so it's good if
            // they're valid identifiers.
            let name = name.chars().enumerate().map(|(i, c)| {
                    if (i == 0 && c.is_xid_start()) ||
                        (i != 0 && c.is_xid_continue()) {
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
        let current_name = match item.name {
            Some(ref name) if !name.is_empty() => Some(name.clone()),
            _ => typename_if_impl(&item)
        };

        let pushed = if let Some(name) = current_name {
            self.names.push(name);
            true
        } else {
            false
        };

        if let Some(doc) = item.doc_value() {
            self.cnt = 0;
            markdown::find_testable_code(doc, &mut *self);
        }

        let ret = self.fold_item_recur(item);
        if pushed {
            self.names.pop();
        }

        return ret;

        // FIXME: it would be better to not have the escaped version in the first place
        fn unescape_for_testname(mut s: String) -> String {
            // for refs `&foo`
            if s.contains("&amp;") {
                s = s.replace("&amp;", "&");

                // `::&'a mut Foo::` looks weird, let's make it `::<&'a mut Foo>`::
                if let Some('&') = s.chars().nth(0) {
                    s = format!("<{}>", s);
                }
            }

            // either `<..>` or `->`
            if s.contains("&gt;") {
                s.replace("&gt;", ">")
                 .replace("&lt;", "<")
            } else {
                s
            }
        }

        fn typename_if_impl(item: &clean::Item) -> Option<String> {
            if let clean::ItemEnum::ImplItem(ref impl_) = item.inner {
                let path = impl_.for_.to_string();
                let unescaped_path = unescape_for_testname(path);
                Some(unescaped_path)
            } else {
                None
            }
        }
    }
}
