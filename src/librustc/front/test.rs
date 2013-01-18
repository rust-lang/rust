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
use session::Session;

use core::dvec::DVec;
use core::option;
use core::vec;
use syntax::ast_util::*;
use syntax::attr;
use syntax::codemap::span;
use syntax::fold;
use syntax::print::pprust;
use syntax::{ast, ast_util};
use syntax::attr::attrs_contains_name;

export modify_for_testing;

type node_id_gen = fn@() -> ast::node_id;

type test = {span: span, path: ~[ast::ident],
             ignore: bool, should_fail: bool};

type test_ctxt =
    @{sess: session::Session,
      crate: @ast::crate,
      mut path: ~[ast::ident],
      testfns: DVec<test>};

// Traverse the crate, collecting all the test functions, eliding any
// existing main functions, and synthesizing a main test harness
fn modify_for_testing(sess: session::Session,
                      crate: @ast::crate) -> @ast::crate {

    // We generate the test harness when building in the 'test'
    // configuration, either with the '--test' or '--cfg test'
    // command line options.
    let should_test = attr::contains(/*bad*/copy crate.node.config,
                                     attr::mk_word_item(~"test"));

    if should_test {
        generate_test_harness(sess, crate)
    } else {
        strip_test_functions(crate)
    }
}

fn generate_test_harness(sess: session::Session,
                         crate: @ast::crate) -> @ast::crate {
    let cx: test_ctxt =
        @{sess: sess,
          crate: crate,
          mut path: ~[],
          testfns: DVec()};

    let precursor = @fold::AstFoldFns {
        fold_crate: fold::wrap(|a,b| fold_crate(cx, a, b) ),
        fold_item: |a,b| fold_item(cx, a, b),
        fold_mod: |a,b| fold_mod(cx, a, b),.. *fold::default_ast_fold()};

    let fold = fold::make_fold(precursor);
    let res = @fold.fold_crate(*crate);
    return res;
}

fn strip_test_functions(crate: @ast::crate) -> @ast::crate {
    // When not compiling with --test we should not compile the
    // #[test] functions
    do config::strip_items(crate) |attrs| {
        !attr::contains_name(attr::attr_metas(attrs), ~"test")
    }
}

fn fold_mod(cx: test_ctxt, m: ast::_mod, fld: fold::ast_fold) -> ast::_mod {

    // Remove any #[main] from the AST so it doesn't clash with
    // the one we're going to add. Only if compiling an executable.

    fn nomain(cx: test_ctxt, item: @ast::item) -> @ast::item {
        if !cx.sess.building_library {
            @ast::item{attrs: item.attrs.filtered(|attr| {
                               attr::get_attr_name(*attr) != ~"main"
                           }),.. copy *item}
        } else { item }
    }

    let mod_nomain = ast::_mod {
        view_items: /*bad*/copy m.view_items,
        items: vec::map(m.items, |i| nomain(cx, *i)),
    };

    fold::noop_fold_mod(mod_nomain, fld)
}

fn fold_crate(cx: test_ctxt, c: ast::crate_, fld: fold::ast_fold) ->
   ast::crate_ {
    let folded = fold::noop_fold_crate(c, fld);

    // Add a special __test module to the crate that will contain code
    // generated for the test harness
    ast::crate_ { module: add_test_module(cx, /*bad*/copy folded.module),
                  .. folded }
}


fn fold_item(cx: test_ctxt, &&i: @ast::item, fld: fold::ast_fold) ->
   Option<@ast::item> {

    cx.path.push(i.ident);
    debug!("current path: %s",
           ast_util::path_name_i(cx.path, cx.sess.parse_sess.interner));

    if is_test_fn(i) {
        match i.node {
          ast::item_fn(_, purity, _, _) if purity == ast::unsafe_fn => {
            cx.sess.span_fatal(
                i.span,
                ~"unsafe functions cannot be used for tests");
          }
          _ => {
            debug!("this is a test function");
            let test = {span: i.span,
                        path: /*bad*/copy cx.path, ignore: is_ignored(cx, i),
                        should_fail: should_fail(i)};
            cx.testfns.push(test);
            debug!("have %u test functions", cx.testfns.len());
          }
        }
    }

    let res = fold::noop_fold_item(i, fld);
    cx.path.pop();
    return res;
}

fn is_test_fn(i: @ast::item) -> bool {
    let has_test_attr = attr::find_attrs_by_name(i.attrs,
                                                 ~"test").is_not_empty();

    fn has_test_signature(i: @ast::item) -> bool {
        match &i.node {
          &ast::item_fn(ref decl, _, ref tps, _) => {
            let no_output = match decl.output.node {
                ast::ty_nil => true,
                _ => false
            };
            decl.inputs.is_empty() && no_output && tps.is_empty()
          }
          _ => false
        }
    }

    return has_test_attr && has_test_signature(i);
}

fn is_ignored(cx: test_ctxt, i: @ast::item) -> bool {
    let ignoreattrs = attr::find_attrs_by_name(i.attrs, "ignore");
    let ignoreitems = attr::attr_metas(ignoreattrs);
    let cfg_metas = vec::concat(vec::filter_map(ignoreitems,
        |i| attr::get_meta_item_list(*i)));
    return if vec::is_not_empty(ignoreitems) {
        config::metas_in_cfg(/*bad*/copy cx.crate.node.config, cfg_metas)
    } else {
        false
    }
}

fn should_fail(i: @ast::item) -> bool {
    vec::len(attr::find_attrs_by_name(i.attrs, ~"should_fail")) > 0u
}

fn add_test_module(cx: test_ctxt, +m: ast::_mod) -> ast::_mod {
    let testmod = mk_test_module(cx);
    ast::_mod {
        items: vec::append_one(/*bad*/copy m.items, testmod),
        .. m
    }
}

/*

We're going to be building a module that looks more or less like:

mod __test {
    #[legacy_exports];

  fn main(args: ~[str]) -> int {
    std::test::test_main(args, tests())
  }

  fn tests() -> ~[std::test::test_desc] {
    ... the list of tests in the crate ...
  }
}

*/

fn mk_test_module(cx: test_ctxt) -> @ast::item {
    // Link to std
    let std = mk_std(cx);
    let view_items = if is_std(cx) { ~[] } else { ~[std] };
    // A function that generates a vector of test descriptors to feed to the
    // test runner
    let testsfn = mk_tests(cx);
    // The synthesized main function which will call the console test runner
    // with our list of tests
    let mainfn = mk_main(cx);
    let testmod = ast::_mod {
        view_items: view_items,
        items: ~[mainfn, testsfn],
    };
    let item_ = ast::item_mod(testmod);
    // This attribute tells resolve to let us call unexported functions
    let resolve_unexported_attr =
        attr::mk_attr(attr::mk_word_item(~"!resolve_unexported"));
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

fn nospan<T: Copy>(t: T) -> ast::spanned<T> {
    ast::spanned { node: t, span: dummy_sp() }
}

fn path_node(+ids: ~[ast::ident]) -> @ast::path {
    @ast::path { span: dummy_sp(),
                 global: false,
                 idents: ids,
                 rp: None,
                 types: ~[] }
}

fn path_node_global(+ids: ~[ast::ident]) -> @ast::path {
    @ast::path { span: dummy_sp(),
                 global: true,
                 idents: ids,
                 rp: None,
                 types: ~[] }
}

fn mk_std(cx: test_ctxt) -> @ast::view_item {
    let vers = ast::lit_str(@~"0.6");
    let vers = nospan(vers);
    let mi = ast::meta_name_value(~"vers", vers);
    let mi = nospan(mi);
    let vi = ast::view_item_use(cx.sess.ident_of(~"std"),
                                ~[@mi],
                                cx.sess.next_node_id());
    let vi = ast::view_item {
        node: vi,
        attrs: ~[],
        vis: ast::private,
        span: dummy_sp()
    };

    return @vi;
}

fn mk_tests(cx: test_ctxt) -> @ast::item {
    let ret_ty = mk_test_desc_vec_ty(cx);

    let decl = ast::fn_decl {
        inputs: ~[],
        output: ret_ty,
        cf: ast::return_val,
    };

    // The vector of test_descs for this crate
    let test_descs = mk_test_desc_vec(cx);

    let body_: ast::blk_ =
        default_block(~[], option::Some(test_descs), cx.sess.next_node_id());
    let body = nospan(body_);

    let item_ = ast::item_fn(decl, ast::impure_fn, ~[], body);
    let item = ast::item {
        ident: cx.sess.ident_of(~"tests"),
        attrs: ~[],
        id: cx.sess.next_node_id(),
        node: item_,
        vis: ast::public,
        span: dummy_sp(),
    };
    return @item;
}

fn is_std(cx: test_ctxt) -> bool {
    let is_std = {
        let items = attr::find_linkage_metas(cx.crate.node.attrs);
        match attr::last_meta_item_value_str_by_name(items, ~"name") {
          Some(~"std") => true,
          _ => false
        }
    };
    return is_std;
}

fn mk_path(cx: test_ctxt, +path: ~[ast::ident]) -> @ast::path {
    // For tests that are inside of std we don't want to prefix
    // the paths with std::
    if is_std(cx) { path_node_global(path) }
    else {
        path_node(
            ~[cx.sess.ident_of(~"self"),
              cx.sess.ident_of(~"std")]
            + path)
    }
}

// The ast::Ty of ~[std::test::test_desc]
fn mk_test_desc_vec_ty(cx: test_ctxt) -> @ast::Ty {
    let test_desc_ty_path =
        mk_path(cx, ~[cx.sess.ident_of(~"test"),
                      cx.sess.ident_of(~"TestDesc")]);

    let test_desc_ty = ast::Ty {
        id: cx.sess.next_node_id(),
        node: ast::ty_path(test_desc_ty_path, cx.sess.next_node_id()),
        span: dummy_sp(),
    };

    let vec_mt = ast::mt {ty: @test_desc_ty, mutbl: ast::m_imm};

    let inner_ty = @ast::Ty {
        id: cx.sess.next_node_id(),
        node: ast::ty_vec(vec_mt),
        span: dummy_sp(),
    };

    @ast::Ty {
        id: cx.sess.next_node_id(),
        node: ast::ty_uniq(ast::mt { ty: inner_ty, mutbl: ast::m_imm }),
        span: dummy_sp(),
    }
}

fn mk_test_desc_vec(cx: test_ctxt) -> @ast::expr {
    debug!("building test vector from %u tests", cx.testfns.len());
    let mut descs = ~[];
    for cx.testfns.each |test| {
        descs.push(mk_test_desc_rec(cx, *test));
    }

    let inner_expr = @ast::expr {
        id: cx.sess.next_node_id(),
        callee_id: cx.sess.next_node_id(),
        node: ast::expr_vec(descs, ast::m_imm),
        span: dummy_sp(),
    };

    @ast::expr {
        id: cx.sess.next_node_id(),
        callee_id: cx.sess.next_node_id(),
        node: ast::expr_vstore(inner_expr, ast::expr_vstore_uniq),
        span: dummy_sp(),
    }
}

fn mk_test_desc_rec(cx: test_ctxt, test: test) -> @ast::expr {
    let span = test.span;
    let path = /*bad*/copy test.path;

    debug!("encoding %s", ast_util::path_name_i(path,
                                                cx.sess.parse_sess.interner));

    let name_lit: ast::lit =
        nospan(ast::lit_str(@ast_util::path_name_i(
            path, cx.sess.parse_sess.interner)));

    let name_expr_inner = @ast::expr {
        id: cx.sess.next_node_id(),
        callee_id: cx.sess.next_node_id(),
        node: ast::expr_lit(@name_lit),
        span: span,
    };

    let name_expr = ast::expr {
        id: cx.sess.next_node_id(),
        callee_id: cx.sess.next_node_id(),
        node: ast::expr_vstore(name_expr_inner, ast::expr_vstore_uniq),
        span: dummy_sp(),
    };

    let name_field = nospan(ast::field_ {
        mutbl: ast::m_imm,
        ident: cx.sess.ident_of(~"name"),
        expr: @name_expr,
    });

    let fn_path = path_node_global(path);

    let fn_expr = ast::expr {
        id: cx.sess.next_node_id(),
        callee_id: cx.sess.next_node_id(),
        node: ast::expr_path(fn_path),
        span: span,
    };

    let fn_wrapper_expr = mk_test_wrapper(cx, fn_expr, span);

    let fn_field = nospan(ast::field_ {
        mutbl: ast::m_imm,
        ident: cx.sess.ident_of(~"testfn"),
        expr: fn_wrapper_expr,
    });

    let ignore_lit: ast::lit = nospan(ast::lit_bool(test.ignore));

    let ignore_expr = ast::expr {
        id: cx.sess.next_node_id(),
        callee_id: cx.sess.next_node_id(),
        node: ast::expr_lit(@ignore_lit),
        span: span,
    };

    let ignore_field = nospan(ast::field_ {
        mutbl: ast::m_imm,
        ident: cx.sess.ident_of(~"ignore"),
        expr: @ignore_expr,
    });

    let fail_lit: ast::lit = nospan(ast::lit_bool(test.should_fail));

    let fail_expr = ast::expr {
        id: cx.sess.next_node_id(),
        callee_id: cx.sess.next_node_id(),
        node: ast::expr_lit(@fail_lit),
        span: span,
    };

    let fail_field = nospan(ast::field_ {
        mutbl: ast::m_imm,
        ident: cx.sess.ident_of(~"should_fail"),
        expr: @fail_expr,
    });

    let test_desc_path =
        mk_path(cx, ~[cx.sess.ident_of(~"test"),
                      cx.sess.ident_of(~"TestDesc")]);

    let desc_rec_ = ast::expr_struct(
        test_desc_path,
        ~[name_field, fn_field, ignore_field, fail_field],
        option::None
    );

    let desc_rec = ast::expr {
        id: cx.sess.next_node_id(),
        callee_id: cx.sess.next_node_id(),
        node: desc_rec_,
        span: span,
    };

    return @desc_rec;
}

// Produces a bare function that wraps the test function

// FIXME (#1281): This can go away once fn is the type of bare function.
fn mk_test_wrapper(cx: test_ctxt,
                   +fn_path_expr: ast::expr,
                   span: span) -> @ast::expr {
    let call_expr = ast::expr {
        id: cx.sess.next_node_id(),
        callee_id: cx.sess.next_node_id(),
        node: ast::expr_call(@fn_path_expr, ~[], false),
        span: span,
    };

    let call_stmt: ast::stmt = nospan(
        ast::stmt_semi(@call_expr, cx.sess.next_node_id()));

    let wrapper_decl = ast::fn_decl {
        inputs: ~[],
        output: @ast::Ty {
            id: cx.sess.next_node_id(),
            node: ast::ty_nil,
            span: span,
        },
        cf: ast::return_val
    };

    let wrapper_body = nospan(ast::blk_ {
        view_items: ~[],
        stmts: ~[@call_stmt],
        expr: option::None,
        id: cx.sess.next_node_id(),
        rules: ast::default_blk
    });

    let wrapper_expr = ast::expr  {
        id: cx.sess.next_node_id(),
        callee_id: cx.sess.next_node_id(),
        node: ast::expr_fn(ast::ProtoBare, wrapper_decl, wrapper_body, @~[]),
        span: span
    };

    return @wrapper_expr;
}

fn mk_main(cx: test_ctxt) -> @ast::item {
    let ret_ty = ast::Ty {
        id: cx.sess.next_node_id(),
        node: ast::ty_nil,
        span: dummy_sp(),
    };

    let decl = ast::fn_decl {
        inputs: ~[],
        output: @ret_ty,
        cf: ast::return_val,
    };

    let test_main_call_expr = mk_test_main_call(cx);

    let body_: ast::blk_ =
        default_block(~[], option::Some(test_main_call_expr),
                      cx.sess.next_node_id());
    let body = ast::spanned { node: body_, span: dummy_sp() };

    let item_ = ast::item_fn(decl, ast::impure_fn, ~[], body);
    let item = ast::item {
        ident: cx.sess.ident_of(~"main"),
        attrs: ~[attr::mk_attr(attr::mk_word_item(~"main"))],
        id: cx.sess.next_node_id(),
        node: item_,
        vis: ast::public,
        span: dummy_sp(),
    };
    return @item;
}

fn mk_test_main_call(cx: test_ctxt) -> @ast::expr {
    // Call os::args to generate the vector of test_descs
    let args_path = path_node_global(~[
        cx.sess.ident_of(~"os"),
        cx.sess.ident_of(~"args")
    ]);

    let args_path_expr = ast::expr {
        id: cx.sess.next_node_id(),
        callee_id: cx.sess.next_node_id(),
        node: ast::expr_path(args_path),
        span: dummy_sp(),
    };

    let args_call_expr = ast::expr {
        id: cx.sess.next_node_id(),
        callee_id: cx.sess.next_node_id(),
        node: ast::expr_call(@args_path_expr, ~[], false),
        span: dummy_sp(),
    };

    // Call __test::test to generate the vector of test_descs
    let test_path = path_node(~[cx.sess.ident_of(~"tests")]);

    let test_path_expr = ast::expr {
        id: cx.sess.next_node_id(),
        callee_id: cx.sess.next_node_id(),
        node: ast::expr_path(test_path),
        span: dummy_sp(),
    };

    let test_call_expr = ast::expr {
        id: cx.sess.next_node_id(),
        callee_id: cx.sess.next_node_id(),
        node: ast::expr_call(@test_path_expr, ~[], false),
        span: dummy_sp(),
    };

    // Call std::test::test_main
    let test_main_path =
        mk_path(cx, ~[cx.sess.ident_of(~"test"),
                      cx.sess.ident_of(~"test_main")]);

    let test_main_path_expr = ast::expr {
        id: cx.sess.next_node_id(),
        callee_id: cx.sess.next_node_id(),
        node: ast::expr_path(test_main_path),
        span: dummy_sp(),
    };

    let test_main_call_expr = ast::expr {
        id: cx.sess.next_node_id(),
        callee_id: cx.sess.next_node_id(),
        node: ast::expr_call(
            @test_main_path_expr,
            ~[@args_call_expr, @test_call_expr],
            false
        ),
        span: dummy_sp(),
    };

    return @test_main_call_expr;
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
