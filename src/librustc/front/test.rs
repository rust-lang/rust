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

use core::prelude::*;

use driver::session;
use front::config;

use core::vec;
use syntax::ast_util::*;
use syntax::attr;
use syntax::codemap::{dummy_sp, span, ExpandedFrom, CallInfo, NameAndSpan};
use syntax::codemap;
use syntax::ext::base::{mk_ctxt, ext_ctxt};
use syntax::fold;
use syntax::print::pprust;
use syntax::{ast, ast_util};

type node_id_gen = @fn() -> ast::node_id;

struct Test {
    span: span,
    path: ~[ast::ident],
    bench: bool,
    ignore: bool,
    should_fail: bool
}

struct TestCtxt {
    sess: session::Session,
    crate: @ast::crate,
    path: ~[ast::ident],
    ext_cx: @ext_ctxt,
    testfns: ~[Test]
}

// Traverse the crate, collecting all the test functions, eliding any
// existing main functions, and synthesizing a main test harness
pub fn modify_for_testing(sess: session::Session,
                          crate: @ast::crate)
                       -> @ast::crate {
    // We generate the test harness when building in the 'test'
    // configuration, either with the '--test' or '--cfg test'
    // command line options.
    let should_test = attr::contains(crate.node.config,
                                     attr::mk_word_item(@~"test"));

    if should_test {
        generate_test_harness(sess, crate)
    } else {
        strip_test_functions(crate)
    }
}

fn generate_test_harness(sess: session::Session,
                         crate: @ast::crate)
                      -> @ast::crate {
    let cx: @mut TestCtxt = @mut TestCtxt {
        sess: sess,
        crate: crate,
        ext_cx: mk_ctxt(sess.parse_sess, copy sess.opts.cfg),
        path: ~[],
        testfns: ~[]
    };

    cx.ext_cx.bt_push(ExpandedFrom(CallInfo {
        call_site: dummy_sp(),
        callee: NameAndSpan {
            name: ~"test",
            span: None
        }
    }));

    let precursor = @fold::AstFoldFns {
        fold_crate: fold::wrap(|a,b| fold_crate(cx, a, b) ),
        fold_item: |a,b| fold_item(cx, a, b),
        fold_mod: |a,b| fold_mod(cx, a, b),.. *fold::default_ast_fold()};

    let fold = fold::make_fold(precursor);
    let res = @fold.fold_crate(&*crate);
    cx.ext_cx.bt_pop();
    return res;
}

fn strip_test_functions(crate: @ast::crate) -> @ast::crate {
    // When not compiling with --test we should not compile the
    // #[test] functions
    do config::strip_items(crate) |attrs| {
        !attr::contains_name(attr::attr_metas(attrs), ~"test") &&
        !attr::contains_name(attr::attr_metas(attrs), ~"bench")
    }
}

fn fold_mod(cx: @mut TestCtxt,
            m: &ast::_mod,
            fld: @fold::ast_fold)
         -> ast::_mod {
    // Remove any #[main] from the AST so it doesn't clash with
    // the one we're going to add. Only if compiling an executable.

    fn nomain(cx: @mut TestCtxt, item: @ast::item) -> @ast::item {
        if !*cx.sess.building_library {
            @ast::item{attrs: item.attrs.filtered(|attr| {
                               *attr::get_attr_name(attr) != ~"main"
                           }),.. copy *item}
        } else { item }
    }

    let mod_nomain = ast::_mod {
        view_items: /*bad*/copy m.view_items,
        items: vec::map(m.items, |i| nomain(cx, *i)),
    };

    fold::noop_fold_mod(&mod_nomain, fld)
}

fn fold_crate(cx: @mut TestCtxt,
              c: &ast::crate_,
              fld: @fold::ast_fold)
           -> ast::crate_ {
    let folded = fold::noop_fold_crate(c, fld);

    // Add a special __test module to the crate that will contain code
    // generated for the test harness
    ast::crate_ {
        module: add_test_module(cx, &folded.module),
        .. folded
    }
}


fn fold_item(cx: @mut TestCtxt, i: @ast::item, fld: @fold::ast_fold)
          -> Option<@ast::item> {
    cx.path.push(i.ident);
    debug!("current path: %s",
           ast_util::path_name_i(copy cx.path, cx.sess.parse_sess.interner));

    if is_test_fn(i) || is_bench_fn(i) {
        match i.node {
          ast::item_fn(_, purity, _, _, _) if purity == ast::unsafe_fn => {
            let sess = cx.sess;
            sess.span_fatal(
                i.span,
                ~"unsafe functions cannot be used for tests");
          }
          _ => {
            debug!("this is a test function");
            let test = Test {
                span: i.span,
                path: /*bad*/copy cx.path,
                bench: is_bench_fn(i),
                ignore: is_ignored(cx, i),
                should_fail: should_fail(i)
            };
            cx.testfns.push(test);
            // debug!("have %u test/bench functions", cx.testfns.len());
          }
        }
    }

    let res = fold::noop_fold_item(i, fld);
    cx.path.pop();
    return res;
}

fn is_test_fn(i: @ast::item) -> bool {
    let has_test_attr = !attr::find_attrs_by_name(i.attrs,
                                                  ~"test").is_empty();

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

    return has_test_attr && has_test_signature(i);
}

fn is_bench_fn(i: @ast::item) -> bool {
    let has_bench_attr =
        vec::len(attr::find_attrs_by_name(i.attrs, ~"bench")) > 0u;

    fn has_test_signature(i: @ast::item) -> bool {
        match i.node {
            ast::item_fn(ref decl, _, _, ref generics, _) => {
                let input_cnt = vec::len(decl.inputs);
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
    let ignoreattrs = attr::find_attrs_by_name(i.attrs, "ignore");
    let ignoreitems = attr::attr_metas(ignoreattrs);
    return if !ignoreitems.is_empty() {
        let cfg_metas =
            vec::concat(
                vec::filter_map(ignoreitems,
                                |i| attr::get_meta_item_list(i)));
        config::metas_in_cfg(/*bad*/copy cx.crate.node.config, cfg_metas)
    } else {
        false
    }
}

fn should_fail(i: @ast::item) -> bool {
    vec::len(attr::find_attrs_by_name(i.attrs, ~"should_fail")) > 0u
}

fn add_test_module(cx: &TestCtxt, m: &ast::_mod) -> ast::_mod {
    let testmod = mk_test_module(cx);
    ast::_mod {
        items: vec::append_one(/*bad*/copy m.items, testmod),
        .. /*bad*/ copy *m
    }
}

/*

We're going to be building a module that looks more or less like:

mod __test {
  #[!resolve_unexported]
  extern mod std (name = "std", vers = "...");
  fn main() {
    #[main];
    std::test::test_main_static(::os::args(), tests)
  }

  static tests : &'static [std::test::TestDescAndFn] = &[
    ... the list of tests in the crate ...
  ];
}

*/

fn mk_std(cx: &TestCtxt) -> @ast::view_item {
    let vers = ast::lit_str(@~"0.7-pre");
    let vers = nospan(vers);
    let mi = ast::meta_name_value(@~"vers", vers);
    let mi = nospan(mi);
    let id_std = cx.sess.ident_of(~"std");
    let vi = if is_std(cx) {
        ast::view_item_use(
            ~[@nospan(ast::view_path_simple(id_std,
                                            path_node(~[id_std]),
                                            ast::type_value_ns,
                                            cx.sess.next_node_id()))])
    } else {
        ast::view_item_extern_mod(id_std, ~[@mi],
                           cx.sess.next_node_id())
    };
    let vi = ast::view_item {
        node: vi,
        attrs: ~[],
        vis: ast::public,
        span: dummy_sp()
    };
    return @vi;
}

fn mk_test_module(cx: &TestCtxt) -> @ast::item {

    // Link to std
    let view_items = ~[mk_std(cx)];

    // A constant vector of test descriptors.
    let tests = mk_tests(cx);

    // The synthesized main function which will call the console test runner
    // with our list of tests
    let ext_cx = cx.ext_cx;
    let mainfn = (quote_item!(
        pub fn main() {
            #[main];
            std::test::test_main_static(::os::args(), tests);
        }
    )).get();

    let testmod = ast::_mod {
        view_items: view_items,
        items: ~[mainfn, tests],
    };
    let item_ = ast::item_mod(testmod);

    // This attribute tells resolve to let us call unexported functions
    let resolve_unexported_attr =
        attr::mk_attr(attr::mk_word_item(@~"!resolve_unexported"));

    let item = ast::item {
        ident: cx.sess.ident_of(~"__test"),
        attrs: ~[resolve_unexported_attr],
        id: cx.sess.next_node_id(),
        node: item_,
        vis: ast::public,
        span: dummy_sp(),
     };

    debug!("Synthetic test module:\n%s\n",
           pprust::item_to_str(@copy item, cx.sess.intr()));

    return @item;
}

fn nospan<T:Copy>(t: T) -> codemap::spanned<T> {
    codemap::spanned { node: t, span: dummy_sp() }
}

fn path_node(ids: ~[ast::ident]) -> @ast::Path {
    @ast::Path { span: dummy_sp(),
                global: false,
                idents: ids,
                rp: None,
                types: ~[] }
}

fn path_node_global(ids: ~[ast::ident]) -> @ast::Path {
    @ast::Path { span: dummy_sp(),
                 global: true,
                 idents: ids,
                 rp: None,
                 types: ~[] }
}

fn mk_tests(cx: &TestCtxt) -> @ast::item {

    let ext_cx = cx.ext_cx;

    // The vector of test_descs for this crate
    let test_descs = mk_test_descs(cx);

    (quote_item!(
        pub static tests : &'static [self::std::test::TestDescAndFn] =
            $test_descs
        ;
    )).get()
}

fn is_std(cx: &TestCtxt) -> bool {
    let is_std = {
        let items = attr::find_linkage_metas(cx.crate.node.attrs);
        match attr::last_meta_item_value_str_by_name(items, ~"name") {
          Some(@~"std") => true,
          _ => false
        }
    };
    return is_std;
}

fn mk_test_descs(cx: &TestCtxt) -> @ast::expr {
    debug!("building test vector from %u tests", cx.testfns.len());
    let mut descs = ~[];
    for cx.testfns.each |test| {
        descs.push(mk_test_desc_and_fn_rec(cx, test));
    }

    let sess = cx.sess;
    let inner_expr = @ast::expr {
        id: sess.next_node_id(),
        callee_id: sess.next_node_id(),
        node: ast::expr_vec(descs, ast::m_imm),
        span: dummy_sp(),
    };

    @ast::expr {
        id: sess.next_node_id(),
        callee_id: sess.next_node_id(),
        node: ast::expr_vstore(inner_expr, ast::expr_vstore_slice),
        span: dummy_sp(),
    }
}

fn mk_test_desc_and_fn_rec(cx: &TestCtxt, test: &Test) -> @ast::expr {
    let span = test.span;
    let path = /*bad*/copy test.path;

    let ext_cx = cx.ext_cx;

    debug!("encoding %s", ast_util::path_name_i(path,
                                                cx.sess.parse_sess.interner));

    let name_lit: ast::lit =
        nospan(ast::lit_str(@ast_util::path_name_i(
            path,
            cx.sess.parse_sess.interner)));

    let name_expr = @ast::expr {
          id: cx.sess.next_node_id(),
          callee_id: cx.sess.next_node_id(),
          node: ast::expr_lit(@name_lit),
          span: span
    };

    let fn_path = path_node_global(path);

    let fn_expr = @ast::expr {
        id: cx.sess.next_node_id(),
        callee_id: cx.sess.next_node_id(),
        node: ast::expr_path(fn_path),
        span: span,
    };

    let t_expr = if test.bench {
        quote_expr!( self::std::test::StaticBenchFn($fn_expr) )
    } else {
        quote_expr!( self::std::test::StaticTestFn($fn_expr) )
    };

    let ignore_expr = if test.ignore {
        quote_expr!( true )
    } else {
        quote_expr!( false )
    };

    let fail_expr = if test.should_fail {
        quote_expr!( true )
    } else {
        quote_expr!( false )
    };

    let e = quote_expr!(
        self::std::test::TestDescAndFn {
            desc: self::std::test::TestDesc {
                name: self::std::test::StaticTestName($name_expr),
                ignore: $ignore_expr,
                should_fail: $fail_expr
            },
            testfn: $t_expr,
        }
    );
    e
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
