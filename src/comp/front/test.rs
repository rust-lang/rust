// Code that generates a test runner to run all the tests in a crate

import core::{option, vec};
import syntax::{ast, ast_util};
import syntax::ast_util::*;
//import syntax::ast_util::dummy_sp;
import syntax::fold;
import syntax::print::pprust;
import syntax::codemap::span;
import driver::session;
import session::session;
import front::attr;

export modify_for_testing;

type node_id_gen = fn@() -> ast::node_id;

type test = {span: span, path: [ast::ident], ignore: bool, should_fail: bool};

type test_ctxt =
    @{sess: session::session,
      crate: @ast::crate,
      mutable path: [ast::ident],
      mutable testfns: [test]};

// Traverse the crate, collecting all the test functions, eliding any
// existing main functions, and synthesizing a main test harness
fn modify_for_testing(sess: session::session,
                      crate: @ast::crate) -> @ast::crate {

    if sess.opts.test {
        generate_test_harness(sess, crate)
    } else {
        strip_test_functions(crate)
    }
}

fn generate_test_harness(sess: session::session,
                         crate: @ast::crate) -> @ast::crate {
    let cx: test_ctxt =
        @{sess: sess,
          crate: crate,
          mutable path: [],
          mutable testfns: []};

    let precursor =
        {fold_crate: fold::wrap(bind fold_crate(cx, _, _)),
         fold_item: bind fold_item(cx, _, _),
         fold_mod: bind fold_mod(cx, _, _) with *fold::default_ast_fold()};

    let fold = fold::make_fold(precursor);
    let res = @fold.fold_crate(*crate);
    ret res;
}

fn strip_test_functions(crate: @ast::crate) -> @ast::crate {
    // When not compiling with --test we should not compile the
    // #[test] functions
    config::strip_items(crate) {|attrs|
        !attr::contains_name(attr::attr_metas(attrs), "test")
    }
}

fn fold_mod(_cx: test_ctxt, m: ast::_mod, fld: fold::ast_fold) -> ast::_mod {

    // Remove any defined main function from the AST so it doesn't clash with
    // the one we're going to add.  FIXME: This is sloppy. Instead we should
    // have some mechanism to indicate to the translation pass which function
    // we want to be main.
    fn nomain(&&item: @ast::item) -> option::t<@ast::item> {
        alt item.node {
          ast::item_fn(_, _, _) {
            if item.ident == "main" {
                option::none
            } else { option::some(item) }
          }
          _ { option::some(item) }
        }
    }

    let mod_nomain =
        {view_items: m.view_items, items: vec::filter_map(m.items, nomain)};
    ret fold::noop_fold_mod(mod_nomain, fld);
}

fn fold_crate(cx: test_ctxt, c: ast::crate_, fld: fold::ast_fold) ->
   ast::crate_ {
    let folded = fold::noop_fold_crate(c, fld);

    // Add a special __test module to the crate that will contain code
    // generated for the test harness
    ret {module: add_test_module(cx, folded.module) with folded};
}


fn fold_item(cx: test_ctxt, &&i: @ast::item, fld: fold::ast_fold) ->
   @ast::item {

    cx.path += [i.ident];
    #debug("current path: %s", ast_util::path_name_i(cx.path));

    if is_test_fn(i) {
        alt i.node {
          ast::item_fn(decl, _, _) if decl.purity == ast::unsafe_fn {
            cx.sess.span_fatal(
                i.span,
                "unsafe functions cannot be used for tests");
          }
          _ {
            #debug("this is a test function");
            let test = {span: i.span,
                        path: cx.path, ignore: is_ignored(cx, i),
                        should_fail: should_fail(i)};
            cx.testfns += [test];
            #debug("have %u test functions", vec::len(cx.testfns));
          }
        }
    }

    let res = fold::noop_fold_item(i, fld);
    vec::pop(cx.path);
    ret res;
}

fn is_test_fn(i: @ast::item) -> bool {
    let has_test_attr =
        vec::len(attr::find_attrs_by_name(i.attrs, "test")) > 0u;

    fn has_test_signature(i: @ast::item) -> bool {
        alt i.node {
          ast::item_fn(decl, tps, _) {
            let input_cnt = vec::len(decl.inputs);
            let no_output = decl.output.node == ast::ty_nil;
            let tparm_cnt = vec::len(tps);
            input_cnt == 0u && no_output && tparm_cnt == 0u
          }
          _ { false }
        }
    }

    ret has_test_attr && has_test_signature(i);
}

fn is_ignored(cx: test_ctxt, i: @ast::item) -> bool {
    let ignoreattrs = attr::find_attrs_by_name(i.attrs, "ignore");
    let ignoreitems = attr::attr_metas(ignoreattrs);
    let cfg_metas = vec::concat(vec::filter_map(ignoreitems,
        {|&&i| attr::get_meta_item_list(i)}));
    ret if vec::is_not_empty(ignoreitems) {
        config::metas_in_cfg(cx.crate.node.config, cfg_metas)
    } else {
        false
    }
}

fn should_fail(i: @ast::item) -> bool {
    vec::len(attr::find_attrs_by_name(i.attrs, "should_fail")) > 0u
}

fn add_test_module(cx: test_ctxt, m: ast::_mod) -> ast::_mod {
    let testmod = mk_test_module(cx);
    ret {items: m.items + [testmod] with m};
}

/*

We're going to be building a module that looks more or less like:

mod __test {

  fn main(args: [str]) -> int {
    std::test::test_main(args, tests())
  }

  fn tests() -> [std::test::test_desc] {
    ... the list of tests in the crate ...
  }
}

*/

fn mk_test_module(cx: test_ctxt) -> @ast::item {
    // A function that generates a vector of test descriptors to feed to the
    // test runner
    let testsfn = mk_tests(cx);
    // The synthesized main function which will call the console test runner
    // with our list of tests
    let mainfn = mk_main(cx);
    let testmod: ast::_mod = {view_items: [], items: [mainfn, testsfn]};
    let item_ = ast::item_mod(testmod);
    // This attribute tells resolve to let us call unexported functions
    let resolve_unexported_attr =
        attr::mk_attr(attr::mk_word_item("!resolve_unexported"));
    let item: ast::item =
        {ident: "__test",
         attrs: [resolve_unexported_attr],
         id: cx.sess.next_node_id(),
         node: item_,
         span: dummy_sp()};

    #debug("Synthetic test module:\n%s\n", pprust::item_to_str(@item));

    ret @item;
}

fn nospan<T: copy>(t: T) -> ast::spanned<T> {
    ret {node: t, span: dummy_sp()};
}

fn mk_tests(cx: test_ctxt) -> @ast::item {
    let ret_ty = mk_test_desc_vec_ty(cx);

    let decl: ast::fn_decl =
        {inputs: [],
         output: ret_ty,
         purity: ast::impure_fn,
         cf: ast::return_val,
         constraints: []};

    // The vector of test_descs for this crate
    let test_descs = mk_test_desc_vec(cx);

    let body_: ast::blk_ =
        default_block([], option::some(test_descs), cx.sess.next_node_id());
    let body = nospan(body_);

    let item_ = ast::item_fn(decl, [], body);
    let item: ast::item =
        {ident: "tests",
         attrs: [],
         id: cx.sess.next_node_id(),
         node: item_,
         span: dummy_sp()};
    ret @item;
}

fn mk_path(cx: test_ctxt, path: [ast::ident]) -> [ast::ident] {
    // For tests that are inside of std we don't want to prefix
    // the paths with std::
    let is_std = {
        let items = attr::find_linkage_metas(cx.crate.node.attrs);
        alt attr::meta_item_value_from_list(items, "name") {
          some("std") { true }
          _ { false }
        }
    };
    (if is_std { [] } else { ["std"] }) + path
}

// The ast::ty of [std::test::test_desc]
fn mk_test_desc_vec_ty(cx: test_ctxt) -> @ast::ty {
    let test_desc_ty_path =
        @nospan({global: false,
                 idents: mk_path(cx, ["test", "test_desc"]),
                 types: []});

    let test_desc_ty: ast::ty =
        nospan(ast::ty_path(test_desc_ty_path, cx.sess.next_node_id()));

    let vec_mt: ast::mt = {ty: @test_desc_ty, mut: ast::imm};

    ret @nospan(ast::ty_vec(vec_mt));
}

fn mk_test_desc_vec(cx: test_ctxt) -> @ast::expr {
    #debug("building test vector from %u tests", vec::len(cx.testfns));
    let descs = [];
    for test: test in cx.testfns {
        let test_ = test; // Satisfy alias analysis
        descs += [mk_test_desc_rec(cx, test_)];
    }

    ret @{id: cx.sess.next_node_id(),
          node: ast::expr_vec(descs, ast::imm),
          span: dummy_sp()};
}

fn mk_test_desc_rec(cx: test_ctxt, test: test) -> @ast::expr {
    let span = test.span;
    let path = test.path;

    #debug("encoding %s", ast_util::path_name_i(path));

    let name_lit: ast::lit =
        nospan(ast::lit_str(ast_util::path_name_i(path)));
    let name_expr: ast::expr =
        {id: cx.sess.next_node_id(),
         node: ast::expr_lit(@name_lit),
         span: span};

    let name_field: ast::field =
        nospan({mut: ast::imm, ident: "name", expr: @name_expr});

    let fn_path = @nospan({global: false, idents: path, types: []});

    let fn_expr: ast::expr =
        {id: cx.sess.next_node_id(),
         node: ast::expr_path(fn_path),
         span: span};

    let fn_wrapper_expr = mk_test_wrapper(cx, fn_expr, span);

    let fn_field: ast::field =
        nospan({mut: ast::imm, ident: "fn", expr: fn_wrapper_expr});

    let ignore_lit: ast::lit = nospan(ast::lit_bool(test.ignore));

    let ignore_expr: ast::expr =
        {id: cx.sess.next_node_id(),
         node: ast::expr_lit(@ignore_lit),
         span: span};

    let ignore_field: ast::field =
        nospan({mut: ast::imm, ident: "ignore", expr: @ignore_expr});

    let fail_lit: ast::lit = nospan(ast::lit_bool(test.should_fail));

    let fail_expr: ast::expr =
        {id: cx.sess.next_node_id(),
         node: ast::expr_lit(@fail_lit),
         span: span};

    let fail_field: ast::field =
        nospan({mut: ast::imm, ident: "should_fail", expr: @fail_expr});

    let desc_rec_: ast::expr_ =
        ast::expr_rec([name_field, fn_field, ignore_field, fail_field],
            option::none);
    let desc_rec: ast::expr =
        {id: cx.sess.next_node_id(), node: desc_rec_, span: span};
    ret @desc_rec;
}

// Produces a bare function that wraps the test function
// FIXME: This can go away once fn is the type of bare function
fn mk_test_wrapper(cx: test_ctxt,
                   fn_path_expr: ast::expr,
                   span: span) -> @ast::expr {
    let call_expr: ast::expr = {
        id: cx.sess.next_node_id(),
        node: ast::expr_call(@fn_path_expr, [], false),
        span: span
    };

    let call_stmt: ast::stmt = nospan(
        ast::stmt_semi(@call_expr, cx.sess.next_node_id()));

    let wrapper_decl: ast::fn_decl = {
        inputs: [],
        output: @nospan(ast::ty_nil),
        purity: ast::impure_fn,
        cf: ast::return_val,
        constraints: []
    };

    let wrapper_body: ast::blk = nospan({
        view_items: [],
        stmts: [@call_stmt],
        expr: option::none,
        id: cx.sess.next_node_id(),
        rules: ast::default_blk
    });

    let wrapper_capture: @ast::capture_clause = @{
        copies: [],
        moves: []
    };

    let wrapper_expr: ast::expr = {
        id: cx.sess.next_node_id(),
        node: ast::expr_fn(ast::proto_bare, wrapper_decl,
                           wrapper_body, wrapper_capture),
        span: span
    };

    ret @wrapper_expr;
}

fn mk_main(cx: test_ctxt) -> @ast::item {

    let args_mt: ast::mt = {ty: @nospan(ast::ty_str), mut: ast::imm};
    let args_ty: ast::ty = nospan(ast::ty_vec(args_mt));

    let args_arg: ast::arg =
        {mode: ast::by_val,
         ty: @args_ty,
         ident: "args",
         id: cx.sess.next_node_id()};

    let ret_ty = nospan(ast::ty_nil);

    let decl: ast::fn_decl =
        {inputs: [args_arg],
         output: @ret_ty,
         purity: ast::impure_fn,
         cf: ast::return_val,
         constraints: []};

    let test_main_call_expr = mk_test_main_call(cx);

    let body_: ast::blk_ =
        default_block([], option::some(test_main_call_expr),
                      cx.sess.next_node_id());
    let body = {node: body_, span: dummy_sp()};

    let item_ = ast::item_fn(decl, [], body);
    let item: ast::item =
        {ident: "main",
         attrs: [],
         id: cx.sess.next_node_id(),
         node: item_,
         span: dummy_sp()};
    ret @item;
}

fn mk_test_main_call(cx: test_ctxt) -> @ast::expr {

    // Get the args passed to main so we can pass the to test_main
    let args_path =
        @nospan({global: false, idents: ["args"], types: []});

    let args_path_expr_: ast::expr_ = ast::expr_path(args_path);

    let args_path_expr: ast::expr =
        {id: cx.sess.next_node_id(), node: args_path_expr_, span: dummy_sp()};

    // Call __test::test to generate the vector of test_descs
    let test_path =
        @nospan({global: false, idents: ["tests"], types: []});

    let test_path_expr_: ast::expr_ = ast::expr_path(test_path);

    let test_path_expr: ast::expr =
        {id: cx.sess.next_node_id(), node: test_path_expr_, span: dummy_sp()};

    let test_call_expr_ = ast::expr_call(@test_path_expr, [], false);

    let test_call_expr: ast::expr =
        {id: cx.sess.next_node_id(), node: test_call_expr_, span: dummy_sp()};

    // Call std::test::test_main
    let test_main_path =
        @nospan({global: false,
                 idents: mk_path(cx, ["test", "test_main"]),
                 types: []});

    let test_main_path_expr_: ast::expr_ = ast::expr_path(test_main_path);

    let test_main_path_expr: ast::expr =
        {id: cx.sess.next_node_id(), node: test_main_path_expr_,
         span: dummy_sp()};

    let test_main_call_expr_: ast::expr_ =
        ast::expr_call(@test_main_path_expr,
                       [@args_path_expr, @test_call_expr], false);

    let test_main_call_expr: ast::expr =
        {id: cx.sess.next_node_id(), node: test_main_call_expr_,
         span: dummy_sp()};

    ret @test_main_call_expr;
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
