// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Code that generates a test runner to run all the tests in a crate


use driver::session;
use front::config;

use std::vec;
use syntax::ast_util::*;
use syntax::attr::AttrMetaMethods;
use syntax::attr;
use syntax::codemap::{dummy_sp, Span, ExpnInfo, NameAndSpan};
use syntax::codemap;
use syntax::ext::base::ExtCtxt;
use syntax::fold::ast_fold;
use syntax::fold;
use syntax::opt_vec;
use syntax::print::pprust;
use syntax::{ast, ast_util};

struct Test {
    span: Span,
    path: ~[ast::Ident],
    bench: bool,
    ignore: bool,
    should_fail: bool
}

struct TestCtxt {
    sess: session::Session,
    path: ~[ast::Ident],
    ext_cx: @ExtCtxt,
    testfns: ~[Test],
    is_extra: bool,
    config: ast::CrateConfig,
}

// Traverse the crate, collecting all the test functions, eliding any
// existing main functions, and synthesizing a main test harness
pub fn modify_for_testing(sess: session::Session,
                          crate: ast::Crate) -> ast::Crate {
    // We generate the test harness when building in the 'test'
    // configuration, either with the '--test' or '--cfg test'
    // command line options.
    let should_test = attr::contains_name(crate.config, "test");

    if should_test {
        generate_test_harness(sess, crate)
    } else {
        strip_test_functions(crate)
    }
}

struct TestHarnessGenerator {
    cx: @mut TestCtxt,
}

impl fold::ast_fold for TestHarnessGenerator {
    fn fold_crate(&self, c: ast::Crate) -> ast::Crate {
        let folded = fold::noop_fold_crate(c, self);

        // Add a special __test module to the crate that will contain code
        // generated for the test harness
        ast::Crate {
            module: add_test_module(self.cx, &folded.module),
            .. folded
        }
    }

    fn fold_item(&self, i: @ast::item) -> Option<@ast::item> {
        self.cx.path.push(i.ident);
        debug!("current path: %s",
               ast_util::path_name_i(self.cx.path.clone()));

        if is_test_fn(self.cx, i) || is_bench_fn(i) {
            match i.node {
                ast::item_fn(_, purity, _, _, _)
                    if purity == ast::unsafe_fn => {
                    let sess = self.cx.sess;
                    sess.span_fatal(i.span,
                                    "unsafe functions cannot be used for \
                                     tests");
                }
                _ => {
                    debug!("this is a test function");
                    let test = Test {
                        span: i.span,
                        path: self.cx.path.clone(),
                        bench: is_bench_fn(i),
                        ignore: is_ignored(self.cx, i),
                        should_fail: should_fail(i)
                    };
                    self.cx.testfns.push(test);
                    // debug!("have %u test/bench functions",
                    //        cx.testfns.len());
                }
            }
        }

        let res = fold::noop_fold_item(i, self);
        self.cx.path.pop();
        return res;
    }

    fn fold_mod(&self, m: &ast::_mod) -> ast::_mod {
        // Remove any #[main] from the AST so it doesn't clash with
        // the one we're going to add. Only if compiling an executable.

        fn nomain(cx: @mut TestCtxt, item: @ast::item) -> @ast::item {
            if !*cx.sess.building_library {
                @ast::item {
                    attrs: do item.attrs.iter().filter_map |attr| {
                        if "main" != attr.name() {
                            Some(*attr)
                        } else {
                            None
                        }
                    }.collect(),
                    .. (*item).clone()
                }
            } else {
                item
            }
        }

        let mod_nomain = ast::_mod {
            view_items: m.view_items.clone(),
            items: m.items.iter().map(|i| nomain(self.cx, *i)).collect(),
        };

        fold::noop_fold_mod(&mod_nomain, self)
    }
}

fn generate_test_harness(sess: session::Session, crate: ast::Crate)
                         -> ast::Crate {
    let cx: @mut TestCtxt = @mut TestCtxt {
        sess: sess,
        ext_cx: ExtCtxt::new(sess.parse_sess, sess.opts.cfg.clone()),
        path: ~[],
        testfns: ~[],
        is_extra: is_extra(&crate),
        config: crate.config.clone(),
    };

    let ext_cx = cx.ext_cx;
    ext_cx.bt_push(ExpnInfo {
        call_site: dummy_sp(),
        callee: NameAndSpan {
            name: @"test",
            span: None
        }
    });

    let fold = TestHarnessGenerator {
        cx: cx
    };
    let res = fold.fold_crate(crate);
    ext_cx.bt_pop();
    return res;
}

fn strip_test_functions(crate: ast::Crate) -> ast::Crate {
    // When not compiling with --test we should not compile the
    // #[test] functions
    do config::strip_items(crate) |attrs| {
        !attr::contains_name(attrs, "test") &&
        !attr::contains_name(attrs, "bench")
    }
}

fn is_test_fn(cx: @mut TestCtxt, i: @ast::item) -> bool {
    let has_test_attr = attr::contains_name(i.attrs, "test");

    fn has_test_signature(i: @ast::item) -> bool {
        match &i.node {
          &ast::item_fn(ref decl, _, _, ref generics, _) => {
            let no_output = match decl.output.node {
                ast::ty_nil => true,
                _ => false
            };
            decl.inputs.is_empty()
                && no_output
                && !generics.is_parameterized()
          }
          _ => false
        }
    }

    if has_test_attr && !has_test_signature(i) {
        let sess = cx.sess;
        sess.span_err(
            i.span,
            "functions used as tests must have signature fn() -> ()."
        );
    }

    return has_test_attr && has_test_signature(i);
}

fn is_bench_fn(i: @ast::item) -> bool {
    let has_bench_attr = attr::contains_name(i.attrs, "bench");

    fn has_test_signature(i: @ast::item) -> bool {
        match i.node {
            ast::item_fn(ref decl, _, _, ref generics, _) => {
                let input_cnt = decl.inputs.len();
                let no_output = match decl.output.node {
                    ast::ty_nil => true,
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

    return has_bench_attr && has_test_signature(i);
}

fn is_ignored(cx: @mut TestCtxt, i: @ast::item) -> bool {
    do i.attrs.iter().any |attr| {
        // check ignore(cfg(foo, bar))
        "ignore" == attr.name() && match attr.meta_item_list() {
            Some(ref cfgs) => attr::test_cfg(cx.config, cfgs.iter().map(|x| *x)),
            None => true
        }
    }
}

fn should_fail(i: @ast::item) -> bool {
    attr::contains_name(i.attrs, "should_fail")
}

fn add_test_module(cx: &TestCtxt, m: &ast::_mod) -> ast::_mod {
    let testmod = mk_test_module(cx);
    ast::_mod {
        items: vec::append_one(m.items.clone(), testmod),
        ..(*m).clone()
    }
}

/*

We're going to be building a module that looks more or less like:

mod __test {
  #[!resolve_unexported]
  extern mod extra (name = "extra", vers = "...");
  fn main() {
    #[main];
    extra::test::test_main_static(::os::args(), tests)
  }

  static tests : &'static [extra::test::TestDescAndFn] = &[
    ... the list of tests in the crate ...
  ];
}

*/

fn mk_std(cx: &TestCtxt) -> ast::view_item {
    let id_extra = cx.sess.ident_of("extra");
    let vi = if cx.is_extra {
        ast::view_item_use(
            ~[@nospan(ast::view_path_simple(id_extra,
                                            path_node(~[id_extra]),
                                            ast::DUMMY_NODE_ID))])
    } else {
        let mi = attr::mk_name_value_item_str(@"vers", @"0.9-pre");
        ast::view_item_extern_mod(id_extra, None, ~[mi], ast::DUMMY_NODE_ID)
    };
    ast::view_item {
        node: vi,
        attrs: ~[],
        vis: ast::public,
        span: dummy_sp()
    }
}

fn mk_test_module(cx: &TestCtxt) -> @ast::item {

    // Link to extra
    let view_items = ~[mk_std(cx)];

    // A constant vector of test descriptors.
    let tests = mk_tests(cx);

    // The synthesized main function which will call the console test runner
    // with our list of tests
    let mainfn = (quote_item!(cx.ext_cx,
        pub fn main() {
            #[main];
            extra::test::test_main_static(::std::os::args(), TESTS);
        }
    )).unwrap();

    let testmod = ast::_mod {
        view_items: view_items,
        items: ~[mainfn, tests],
    };
    let item_ = ast::item_mod(testmod);

    // This attribute tells resolve to let us call unexported functions
    let resolve_unexported_attr =
        attr::mk_attr(attr::mk_word_item(@"!resolve_unexported"));

    let item = ast::item {
        ident: cx.sess.ident_of("__test"),
        attrs: ~[resolve_unexported_attr],
        id: ast::DUMMY_NODE_ID,
        node: item_,
        vis: ast::public,
        span: dummy_sp(),
     };

    debug!("Synthetic test module:\n%s\n",
           pprust::item_to_str(@item.clone(), cx.sess.intr()));

    return @item;
}

fn nospan<T>(t: T) -> codemap::Spanned<T> {
    codemap::Spanned { node: t, span: dummy_sp() }
}

fn path_node(ids: ~[ast::Ident]) -> ast::Path {
    ast::Path {
        span: dummy_sp(),
        global: false,
        segments: ids.move_iter().map(|identifier| ast::PathSegment {
            identifier: identifier,
            lifetime: None,
            types: opt_vec::Empty,
        }).collect()
    }
}

fn path_node_global(ids: ~[ast::Ident]) -> ast::Path {
    ast::Path {
        span: dummy_sp(),
        global: true,
        segments: ids.move_iter().map(|identifier| ast::PathSegment {
            identifier: identifier,
            lifetime: None,
            types: opt_vec::Empty,
        }).collect()
    }
}

fn mk_tests(cx: &TestCtxt) -> @ast::item {
    // The vector of test_descs for this crate
    let test_descs = mk_test_descs(cx);

    (quote_item!(cx.ext_cx,
        pub static TESTS : &'static [self::extra::test::TestDescAndFn] =
            $test_descs
        ;
    )).unwrap()
}

fn is_extra(crate: &ast::Crate) -> bool {
    let items = attr::find_linkage_metas(crate.attrs);
    match attr::last_meta_item_value_str_by_name(items, "name") {
        Some(s) if "extra" == s => true,
        _ => false
    }
}

fn mk_test_descs(cx: &TestCtxt) -> @ast::Expr {
    debug!("building test vector from %u tests", cx.testfns.len());
    let mut descs = ~[];
    for test in cx.testfns.iter() {
        descs.push(mk_test_desc_and_fn_rec(cx, test));
    }

    let inner_expr = @ast::Expr {
        id: ast::DUMMY_NODE_ID,
        node: ast::ExprVec(descs, ast::MutImmutable),
        span: dummy_sp(),
    };

    @ast::Expr {
        id: ast::DUMMY_NODE_ID,
        node: ast::ExprVstore(inner_expr, ast::ExprVstoreSlice),
        span: dummy_sp(),
    }
}

fn mk_test_desc_and_fn_rec(cx: &TestCtxt, test: &Test) -> @ast::Expr {
    let span = test.span;
    let path = test.path.clone();

    debug!("encoding %s", ast_util::path_name_i(path));

    let name_lit: ast::lit =
        nospan(ast::lit_str(ast_util::path_name_i(path).to_managed()));

    let name_expr = @ast::Expr {
          id: ast::DUMMY_NODE_ID,
          node: ast::ExprLit(@name_lit),
          span: span
    };

    let fn_path = path_node_global(path);

    let fn_expr = @ast::Expr {
        id: ast::DUMMY_NODE_ID,
        node: ast::ExprPath(fn_path),
        span: span,
    };

    let t_expr = if test.bench {
        quote_expr!(cx.ext_cx, self::extra::test::StaticBenchFn($fn_expr) )
    } else {
        quote_expr!(cx.ext_cx, self::extra::test::StaticTestFn($fn_expr) )
    };

    let ignore_expr = if test.ignore {
        quote_expr!(cx.ext_cx, true )
    } else {
        quote_expr!(cx.ext_cx, false )
    };

    let fail_expr = if test.should_fail {
        quote_expr!(cx.ext_cx, true )
    } else {
        quote_expr!(cx.ext_cx, false )
    };

    let e = quote_expr!(cx.ext_cx,
        self::extra::test::TestDescAndFn {
            desc: self::extra::test::TestDesc {
                name: self::extra::test::StaticTestName($name_expr),
                ignore: $ignore_expr,
                should_fail: $fail_expr
            },
            testfn: $t_expr,
        }
    );
    e
}
