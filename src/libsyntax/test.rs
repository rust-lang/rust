// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Code that generates a test runner to run all the tests in a crate

#![allow(dead_code)]
#![allow(unused_imports)]

use std::slice;
use std::mem;
use std::vec;
use ast_util::*;
use attr::AttrMetaMethods;
use attr;
use codemap::{DUMMY_SP, Span, ExpnInfo, NameAndSpan, MacroAttribute};
use codemap;
use diagnostic;
use config;
use ext::base::ExtCtxt;
use ext::build::AstBuilder;
use ext::expand::ExpansionConfig;
use fold::{Folder, MoveMap};
use fold;
use owned_slice::OwnedSlice;
use parse::token::InternedString;
use parse::{token, ParseSess};
use print::pprust;
use {ast, ast_util};
use ptr::P;
use util::small_vector::SmallVector;

struct Test {
    span: Span,
    path: Vec<ast::Ident> ,
    bench: bool,
    ignore: bool,
    should_fail: bool
}

struct TestCtxt<'a> {
    sess: &'a ParseSess,
    span_diagnostic: &'a diagnostic::SpanHandler,
    path: Vec<ast::Ident>,
    ext_cx: ExtCtxt<'a>,
    testfns: Vec<Test>,
    reexport_test_harness_main: Option<InternedString>,
    is_test_crate: bool,
    config: ast::CrateConfig,

    // top-level re-export submodule, filled out after folding is finished
    toplevel_reexport: Option<ast::Ident>,
}

// Traverse the crate, collecting all the test functions, eliding any
// existing main functions, and synthesizing a main test harness
pub fn modify_for_testing(sess: &ParseSess,
                          cfg: &ast::CrateConfig,
                          krate: ast::Crate,
                          span_diagnostic: &diagnostic::SpanHandler) -> ast::Crate {
    // We generate the test harness when building in the 'test'
    // configuration, either with the '--test' or '--cfg test'
    // command line options.
    let should_test = attr::contains_name(krate.config.as_slice(), "test");

    // Check for #[reexport_test_harness_main = "some_name"] which
    // creates a `use some_name = __test::main;`. This needs to be
    // unconditional, so that the attribute is still marked as used in
    // non-test builds.
    let reexport_test_harness_main =
        attr::first_attr_value_str_by_name(krate.attrs.as_slice(),
                                           "reexport_test_harness_main");

    if should_test {
        generate_test_harness(sess, reexport_test_harness_main, krate, cfg, span_diagnostic)
    } else {
        strip_test_functions(krate)
    }
}

struct TestHarnessGenerator<'a> {
    cx: TestCtxt<'a>,
    tests: Vec<ast::Ident>,

    // submodule name, gensym'd identifier for re-exports
    tested_submods: Vec<(ast::Ident, ast::Ident)>,
}

impl<'a> fold::Folder for TestHarnessGenerator<'a> {
    fn fold_crate(&mut self, c: ast::Crate) -> ast::Crate {
        let mut folded = fold::noop_fold_crate(c, self);

        // Add a special __test module to the crate that will contain code
        // generated for the test harness
        let (mod_, reexport) = mk_test_module(&mut self.cx);
        folded.module.items.push(mod_);
        match reexport {
            Some(re) => folded.module.view_items.push(re),
            None => {}
        }
        folded
    }

    fn fold_item(&mut self, i: P<ast::Item>) -> SmallVector<P<ast::Item>> {
        self.cx.path.push(i.ident);
        debug!("current path: {}",
               ast_util::path_name_i(self.cx.path.as_slice()));

        if is_test_fn(&self.cx, &*i) || is_bench_fn(&self.cx, &*i) {
            match i.node {
                ast::ItemFn(_, ast::UnsafeFn, _, _, _) => {
                    let diag = self.cx.span_diagnostic;
                    diag.span_fatal(i.span,
                                    "unsafe functions cannot be used for \
                                     tests");
                }
                _ => {
                    debug!("this is a test function");
                    let test = Test {
                        span: i.span,
                        path: self.cx.path.clone(),
                        bench: is_bench_fn(&self.cx, &*i),
                        ignore: is_ignored(&self.cx, &*i),
                        should_fail: should_fail(&*i)
                    };
                    self.cx.testfns.push(test);
                    self.tests.push(i.ident);
                    // debug!("have {} test/bench functions",
                    //        cx.testfns.len());
                }
            }
        }

        // We don't want to recurse into anything other than mods, since
        // mods or tests inside of functions will break things
        let res = match i.node {
            ast::ItemMod(..) => fold::noop_fold_item(i, self),
            _ => SmallVector::one(i),
        };
        self.cx.path.pop();
        res
    }

    fn fold_mod(&mut self, m: ast::Mod) -> ast::Mod {
        let tests = mem::replace(&mut self.tests, Vec::new());
        let tested_submods = mem::replace(&mut self.tested_submods, Vec::new());
        let mut mod_folded = fold::noop_fold_mod(m, self);
        let tests = mem::replace(&mut self.tests, tests);
        let tested_submods = mem::replace(&mut self.tested_submods, tested_submods);

        // Remove any #[main] from the AST so it doesn't clash with
        // the one we're going to add. Only if compiling an executable.

        mod_folded.items = mem::replace(&mut mod_folded.items, vec![]).move_map(|item| {
            item.map(|ast::Item {id, ident, attrs, node, vis, span}| {
                ast::Item {
                    id: id,
                    ident: ident,
                    attrs: attrs.into_iter().filter_map(|attr| {
                        if !attr.check_name("main") {
                            Some(attr)
                        } else {
                            None
                        }
                    }).collect(),
                    node: node,
                    vis: vis,
                    span: span
                }
            })
        });

        if !tests.is_empty() || !tested_submods.is_empty() {
            let (it, sym) = mk_reexport_mod(&mut self.cx, tests, tested_submods);
            mod_folded.items.push(it);

            if !self.cx.path.is_empty() {
                self.tested_submods.push((self.cx.path[self.cx.path.len()-1], sym));
            } else {
                debug!("pushing nothing, sym: {}", sym);
                self.cx.toplevel_reexport = Some(sym);
            }
        }

        mod_folded
    }
}

fn mk_reexport_mod(cx: &mut TestCtxt, tests: Vec<ast::Ident>,
                   tested_submods: Vec<(ast::Ident, ast::Ident)>) -> (P<ast::Item>, ast::Ident) {
    let mut view_items = Vec::new();
    let super_ = token::str_to_ident("super");

    view_items.extend(tests.into_iter().map(|r| {
        cx.ext_cx.view_use_simple(DUMMY_SP, ast::Public,
                                  cx.ext_cx.path(DUMMY_SP, vec![super_, r]))
    }));
    view_items.extend(tested_submods.into_iter().map(|(r, sym)| {
        let path = cx.ext_cx.path(DUMMY_SP, vec![super_, r, sym]);
        cx.ext_cx.view_use_simple_(DUMMY_SP, ast::Public, r, path)
    }));

    let reexport_mod = ast::Mod {
        inner: DUMMY_SP,
        view_items: view_items,
        items: Vec::new(),
    };

    let sym = token::gensym_ident("__test_reexports");
    let it = P(ast::Item {
        ident: sym.clone(),
        attrs: Vec::new(),
        id: ast::DUMMY_NODE_ID,
        node: ast::ItemMod(reexport_mod),
        vis: ast::Public,
        span: DUMMY_SP,
    });

    (it, sym)
}

fn generate_test_harness(sess: &ParseSess,
                         reexport_test_harness_main: Option<InternedString>,
                         krate: ast::Crate,
                         cfg: &ast::CrateConfig,
                         sd: &diagnostic::SpanHandler) -> ast::Crate {
    let mut cx: TestCtxt = TestCtxt {
        sess: sess,
        span_diagnostic: sd,
        ext_cx: ExtCtxt::new(sess, cfg.clone(),
                             ExpansionConfig::default("test".to_string())),
        path: Vec::new(),
        testfns: Vec::new(),
        reexport_test_harness_main: reexport_test_harness_main,
        is_test_crate: is_test_crate(&krate),
        config: krate.config.clone(),
        toplevel_reexport: None,
    };

    cx.ext_cx.bt_push(ExpnInfo {
        call_site: DUMMY_SP,
        callee: NameAndSpan {
            name: "test".to_string(),
            format: MacroAttribute,
            span: None
        }
    });

    let mut fold = TestHarnessGenerator {
        cx: cx,
        tests: Vec::new(),
        tested_submods: Vec::new(),
    };
    let res = fold.fold_crate(krate);
    fold.cx.ext_cx.bt_pop();
    return res;
}

fn strip_test_functions(krate: ast::Crate) -> ast::Crate {
    // When not compiling with --test we should not compile the
    // #[test] functions
    config::strip_items(krate, |attrs| {
        !attr::contains_name(attrs.as_slice(), "test") &&
        !attr::contains_name(attrs.as_slice(), "bench")
    })
}

fn is_test_fn(cx: &TestCtxt, i: &ast::Item) -> bool {
    let has_test_attr = attr::contains_name(i.attrs.as_slice(), "test");

    #[deriving(PartialEq)]
    enum HasTestSignature {
        Yes,
        No,
        NotEvenAFunction,
    }

    fn has_test_signature(i: &ast::Item) -> HasTestSignature {
        match &i.node {
          &ast::ItemFn(ref decl, _, _, ref generics, _) => {
            let no_output = match decl.output.node {
                ast::TyNil => true,
                _ => false,
            };
            if decl.inputs.is_empty()
                   && no_output
                   && !generics.is_parameterized() {
                Yes
            } else {
                No
            }
          }
          _ => NotEvenAFunction,
        }
    }

    if has_test_attr {
        let diag = cx.span_diagnostic;
        match has_test_signature(i) {
            Yes => {},
            No => diag.span_err(i.span, "functions used as tests must have signature fn() -> ()"),
            NotEvenAFunction => diag.span_err(i.span,
                                              "only functions may be used as tests"),
        }
    }

    return has_test_attr && has_test_signature(i) == Yes;
}

fn is_bench_fn(cx: &TestCtxt, i: &ast::Item) -> bool {
    let has_bench_attr = attr::contains_name(i.attrs.as_slice(), "bench");

    fn has_test_signature(i: &ast::Item) -> bool {
        match i.node {
            ast::ItemFn(ref decl, _, _, ref generics, _) => {
                let input_cnt = decl.inputs.len();
                let no_output = match decl.output.node {
                    ast::TyNil => true,
                    _ => false
                };
                let tparm_cnt = generics.ty_params.len();
                // NB: inadequate check, but we're running
                // well before resolve, can't get too deep.
                input_cnt == 1u
                    && no_output && tparm_cnt == 0u
            }
          _ => false
        }
    }

    if has_bench_attr && !has_test_signature(i) {
        let diag = cx.span_diagnostic;
        diag.span_err(i.span, "functions used as benches must have signature \
                      `fn(&mut Bencher) -> ()`");
    }

    return has_bench_attr && has_test_signature(i);
}

fn is_ignored(cx: &TestCtxt, i: &ast::Item) -> bool {
    i.attrs.iter().any(|attr| {
        // check ignore(cfg(foo, bar))
        attr.check_name("ignore") && match attr.meta_item_list() {
            Some(ref cfgs) => {
                if cfgs.iter().any(|cfg| cfg.check_name("cfg")) {
                    cx.span_diagnostic.span_warn(attr.span,
                            "The use of cfg filters in #[ignore] is \
                             deprecated. Use #[cfg_attr(<cfg pattern>, \
                             ignore)] instead.");
                }
                attr::test_cfg(cx.config.as_slice(), cfgs.iter())
            }
            None => true
        }
    })
}

fn should_fail(i: &ast::Item) -> bool {
    attr::contains_name(i.attrs.as_slice(), "should_fail")
}

/*

We're going to be building a module that looks more or less like:

mod __test {
  extern crate test (name = "test", vers = "...");
  fn main() {
    test::test_main_static(::os::args().as_slice(), tests)
  }

  static tests : &'static [test::TestDescAndFn] = &[
    ... the list of tests in the crate ...
  ];
}

*/

fn mk_std(cx: &TestCtxt) -> ast::ViewItem {
    let id_test = token::str_to_ident("test");
    let (vi, vis) = if cx.is_test_crate {
        (ast::ViewItemUse(
            P(nospan(ast::ViewPathSimple(id_test,
                                         path_node(vec!(id_test)),
                                         ast::DUMMY_NODE_ID)))),
         ast::Public)
    } else {
        (ast::ViewItemExternCrate(id_test, None, ast::DUMMY_NODE_ID),
         ast::Inherited)
    };
    ast::ViewItem {
        node: vi,
        attrs: Vec::new(),
        vis: vis,
        span: DUMMY_SP
    }
}

fn mk_test_module(cx: &mut TestCtxt) -> (P<ast::Item>, Option<ast::ViewItem>) {
    // Link to test crate
    let view_items = vec!(mk_std(cx));

    // A constant vector of test descriptors.
    let tests = mk_tests(cx);

    // The synthesized main function which will call the console test runner
    // with our list of tests
    let mainfn = (quote_item!(&mut cx.ext_cx,
        pub fn main() {
            #![main]
            use std::slice::Slice;
            test::test_main_static(::std::os::args().as_slice(), TESTS);
        }
    )).unwrap();

    let testmod = ast::Mod {
        inner: DUMMY_SP,
        view_items: view_items,
        items: vec!(mainfn, tests),
    };
    let item_ = ast::ItemMod(testmod);

    let mod_ident = token::gensym_ident("__test");
    let item = ast::Item {
        ident: mod_ident,
        attrs: Vec::new(),
        id: ast::DUMMY_NODE_ID,
        node: item_,
        vis: ast::Public,
        span: DUMMY_SP,
    };
    let reexport = cx.reexport_test_harness_main.as_ref().map(|s| {
        // building `use <ident> = __test::main`
        let reexport_ident = token::str_to_ident(s.get());

        let use_path =
            nospan(ast::ViewPathSimple(reexport_ident,
                                       path_node(vec![mod_ident, token::str_to_ident("main")]),
                                       ast::DUMMY_NODE_ID));

        ast::ViewItem {
            node: ast::ViewItemUse(P(use_path)),
            attrs: vec![],
            vis: ast::Inherited,
            span: DUMMY_SP
        }
    });

    debug!("Synthetic test module:\n{}\n", pprust::item_to_string(&item));

    (P(item), reexport)
}

fn nospan<T>(t: T) -> codemap::Spanned<T> {
    codemap::Spanned { node: t, span: DUMMY_SP }
}

fn path_node(ids: Vec<ast::Ident> ) -> ast::Path {
    ast::Path {
        span: DUMMY_SP,
        global: false,
        segments: ids.into_iter().map(|identifier| ast::PathSegment {
            identifier: identifier,
            lifetimes: Vec::new(),
            types: OwnedSlice::empty(),
        }).collect()
    }
}

fn mk_tests(cx: &TestCtxt) -> P<ast::Item> {
    // The vector of test_descs for this crate
    let test_descs = mk_test_descs(cx);

    // FIXME #15962: should be using quote_item, but that stringifies
    // __test_reexports, causing it to be reinterned, losing the
    // gensym information.
    let sp = DUMMY_SP;
    let ecx = &cx.ext_cx;
    let struct_type = ecx.ty_path(ecx.path(sp, vec![ecx.ident_of("self"),
                                                    ecx.ident_of("test"),
                                                    ecx.ident_of("TestDescAndFn")]),
                                  None);
    let static_lt = ecx.lifetime(sp, token::special_idents::static_lifetime.name);
    // &'static [self::test::TestDescAndFn]
    let static_type = ecx.ty_rptr(sp,
                                  ecx.ty(sp, ast::TyVec(struct_type)),
                                  Some(static_lt),
                                  ast::MutImmutable);
    // static TESTS: $static_type = &[...];
    ecx.item_static(sp,
                    ecx.ident_of("TESTS"),
                    static_type,
                    ast::MutImmutable,
                    test_descs)
}

fn is_test_crate(krate: &ast::Crate) -> bool {
    match attr::find_crate_name(krate.attrs.as_slice()) {
        Some(ref s) if "test" == s.get().as_slice() => true,
        _ => false
    }
}

fn mk_test_descs(cx: &TestCtxt) -> P<ast::Expr> {
    debug!("building test vector from {} tests", cx.testfns.len());

    P(ast::Expr {
        id: ast::DUMMY_NODE_ID,
        node: ast::ExprAddrOf(ast::MutImmutable,
            P(ast::Expr {
                id: ast::DUMMY_NODE_ID,
                node: ast::ExprVec(cx.testfns.iter().map(|test| {
                    mk_test_desc_and_fn_rec(cx, test)
                }).collect()),
                span: DUMMY_SP,
            })),
        span: DUMMY_SP,
    })
}

fn mk_test_desc_and_fn_rec(cx: &TestCtxt, test: &Test) -> P<ast::Expr> {
    // FIXME #15962: should be using quote_expr, but that stringifies
    // __test_reexports, causing it to be reinterned, losing the
    // gensym information.

    let span = test.span;
    let path = test.path.clone();
    let ecx = &cx.ext_cx;
    let self_id = ecx.ident_of("self");
    let test_id = ecx.ident_of("test");

    // creates self::test::$name
    let test_path = |name| {
        ecx.path(span, vec![self_id, test_id, ecx.ident_of(name)])
    };
    // creates $name: $expr
    let field = |name, expr| ecx.field_imm(span, ecx.ident_of(name), expr);

    debug!("encoding {}", ast_util::path_name_i(path.as_slice()));

    // path to the #[test] function: "foo::bar::baz"
    let path_string = ast_util::path_name_i(path.as_slice());
    let name_expr = ecx.expr_str(span, token::intern_and_get_ident(path_string.as_slice()));

    // self::test::StaticTestName($name_expr)
    let name_expr = ecx.expr_call(span,
                                  ecx.expr_path(test_path("StaticTestName")),
                                  vec![name_expr]);

    let ignore_expr = ecx.expr_bool(span, test.ignore);
    let fail_expr = ecx.expr_bool(span, test.should_fail);

    // self::test::TestDesc { ... }
    let desc_expr = ecx.expr_struct(
        span,
        test_path("TestDesc"),
        vec![field("name", name_expr),
             field("ignore", ignore_expr),
             field("should_fail", fail_expr)]);


    let mut visible_path = match cx.toplevel_reexport {
        Some(id) => vec![id],
        None => {
            let diag = cx.span_diagnostic;
            diag.handler.bug("expected to find top-level re-export name, but found None");
        }
    };
    visible_path.extend(path.into_iter());

    let fn_expr = ecx.expr_path(ecx.path_global(span, visible_path));

    let variant_name = if test.bench { "StaticBenchFn" } else { "StaticTestFn" };
    // self::test::$variant_name($fn_expr)
    let testfn_expr = ecx.expr_call(span, ecx.expr_path(test_path(variant_name)), vec![fn_expr]);

    // self::test::TestDescAndFn { ... }
    ecx.expr_struct(span,
                    test_path("TestDescAndFn"),
                    vec![field("desc", desc_expr),
                         field("testfn", testfn_expr)])
}
