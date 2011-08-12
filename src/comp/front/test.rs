// Code that generates a test runner to run all the tests in a crate

import std::option;
import std::ivec;
import syntax::ast;
import syntax::fold;
import syntax::print::pprust;
import front::attr;

export modify_for_testing;

type node_id_gen = @fn() -> ast::node_id ;

type test = {path: [ast::ident], ignore: bool};

type test_ctxt =
    @{next_node_id: node_id_gen,
      mutable path: [ast::ident],
      mutable testfns: [test]};

// Traverse the crate, collecting all the test functions, eliding any
// existing main functions, and synthesizing a main test harness
fn modify_for_testing(crate: @ast::crate) -> @ast::crate {

    // FIXME: This hackasaurus assumes that 200000 is a safe number to start
    // generating node_ids at (which is totally not the case). pauls is going
    // to land a patch that puts parse_sess into session, which will give us
    // access to the real next node_id.
    let next_node_id = @mutable 200000;
    let next_node_id_fn =
        @bind fn (next_node_id: @mutable ast::node_id) -> ast::node_id {
                  let this_node_id = *next_node_id;
                  *next_node_id += 1;
                  ret this_node_id;
              }(next_node_id);

    let cx: test_ctxt =
        @{next_node_id: next_node_id_fn,
          mutable path: ~[],
          mutable testfns: ~[]};

    let precursor =
        {fold_crate: bind fold_crate(cx, _, _),
         fold_item: bind fold_item(cx, _, _),
         fold_mod: bind fold_mod(cx, _, _) with *fold::default_ast_fold()};

    let fold = fold::make_fold(precursor);
    let res = @fold.fold_crate(*crate);
    // FIXME: This is necessary to break a circular reference
    fold::dummy_out(fold);
    ret res;
}

fn fold_mod(cx: &test_ctxt, m: &ast::_mod, fld: fold::ast_fold) -> ast::_mod {

    // Remove any defined main function from the AST so it doesn't clash with
    // the one we're going to add.  FIXME: This is sloppy. Instead we should
    // have some mechanism to indicate to the translation pass which function
    // we want to be main.
    fn nomain(item: &@ast::item) -> option::t[@ast::item] {
        alt item.node {
          ast::item_fn(f, _) {
            if item.ident == "main" {
                option::none
            } else { option::some(item) }
          }
          _ { option::some(item) }
        }
    }

    let mod_nomain =
        {view_items: m.view_items, items: ivec::filter_map(nomain, m.items)};
    ret fold::noop_fold_mod(mod_nomain, fld);
}

fn fold_crate(cx: &test_ctxt, c: &ast::crate_, fld: fold::ast_fold) ->
   ast::crate_ {
    let folded = fold::noop_fold_crate(c, fld);

    // Add a special __test module to the crate that will contain code
    // generated for the test harness
    ret {module: add_test_module(cx, folded.module) with folded};
}


fn fold_item(cx: &test_ctxt, i: &@ast::item, fld: fold::ast_fold) ->
   @ast::item {

    cx.path += ~[i.ident];
    log #fmt("current path: %s", ast::path_name_i(cx.path));

    if is_test_fn(i) {
        log "this is a test function";
        let test = {path: cx.path, ignore: is_ignored(i)};
        cx.testfns += ~[test];
        log #fmt("have %u test functions", ivec::len(cx.testfns));
    }

    let res = fold::noop_fold_item(i, fld);
    ivec::pop(cx.path);
    ret res;
}

fn is_test_fn(i: &@ast::item) -> bool {
    let has_test_attr =
        ivec::len(attr::find_attrs_by_name(i.attrs, "test")) > 0u;

    fn has_test_signature(i: &@ast::item) -> bool {
        alt i.node {
          ast::item_fn(f, tps) {
            let input_cnt = ivec::len(f.decl.inputs);
            let no_output = f.decl.output.node == ast::ty_nil;
            let tparm_cnt = ivec::len(tps);
            input_cnt == 0u && no_output && tparm_cnt == 0u
          }
          _ { false }
        }
    }

    ret has_test_attr && has_test_signature(i);
}

fn is_ignored(i: &@ast::item) -> bool {
    attr::contains_name(attr::attr_metas(i.attrs), "ignore")
}

fn add_test_module(cx: &test_ctxt, m: &ast::_mod) -> ast::_mod {
    let testmod = mk_test_module(cx);
    ret {items: m.items + ~[testmod] with m};
}

/*

We're going to be building a module that looks more or less like:

mod __test {

  fn main(vec[str] args) -> int {
    std::test::test_main(args, tests())
  }

  fn tests() -> [std::test::test_desc] {
    ... the list of tests in the crate ...
  }
}

*/

fn mk_test_module(cx: &test_ctxt) -> @ast::item {
    // A function that generates a vector of test descriptors to feed to the
    // test runner
    let testsfn = mk_tests(cx);
    // The synthesized main function which will call the console test runner
    // with our list of tests
    let mainfn = mk_main(cx);
    let testmod: ast::_mod = {view_items: ~[], items: ~[mainfn, testsfn]};
    let item_ = ast::item_mod(testmod);
    let item: ast::item =
        {ident: "__test",
         attrs: ~[],
         id: cx.next_node_id(),
         node: item_,
         span: ast::dummy_sp()};

    log #fmt("Synthetic test module:\n%s\n", pprust::item_to_str(@item));

    ret @item;
}

fn nospan[T](t: &T) -> ast::spanned[T] {
    ret {node: t, span: ast::dummy_sp()};
}

fn mk_tests(cx: &test_ctxt) -> @ast::item {
    let ret_ty = mk_test_desc_ivec_ty(cx);

    let decl: ast::fn_decl =
        {inputs: ~[],
         output: ret_ty,
         purity: ast::impure_fn,
         il: ast::il_normal,
         cf: ast::return,
         constraints: ~[]};
    let proto = ast::proto_fn;

    // The vector of test_descs for this crate
    let test_descs = mk_test_desc_vec(cx);

    let body_: ast::blk_ =
        {stmts: ~[], expr: option::some(test_descs), id: cx.next_node_id()};
    let body = nospan(body_);

    let fn_ = {decl: decl, proto: proto, body: body};

    let item_ = ast::item_fn(fn_, ~[]);
    let item: ast::item =
        {ident: "tests",
         attrs: ~[],
         id: cx.next_node_id(),
         node: item_,
         span: ast::dummy_sp()};
    ret @item;
}

fn empty_fn_ty() -> ast::ty {
    let proto = ast::proto_fn;
    let input_ty = ~[];
    let ret_ty = @nospan(ast::ty_nil);
    let cf = ast::return;
    let constrs = ~[];
    ret nospan(ast::ty_fn(proto, input_ty, ret_ty, cf, constrs));
}

// The ast::ty of [std::test::test_desc]
fn mk_test_desc_ivec_ty(cx: &test_ctxt) -> @ast::ty {
    let test_desc_ty_path: ast::path =
        nospan({global: false,
                idents: ~["std", "test", "test_desc"],
                types: ~[]});

    let test_desc_ty: ast::ty =
        nospan(ast::ty_path(test_desc_ty_path, cx.next_node_id()));

    let ivec_mt: ast::mt = {ty: @test_desc_ty, mut: ast::imm};

    ret @nospan(ast::ty_ivec(ivec_mt));
}

fn mk_test_desc_vec(cx: &test_ctxt) -> @ast::expr {
    log #fmt("building test vector from %u tests", ivec::len(cx.testfns));
    let descs = ~[];
    for test: test  in cx.testfns {
        let test_ = test; // Satisfy alias analysis
        descs += ~[mk_test_desc_rec(cx, test_)];
    }

    ret @{id: cx.next_node_id(),
          node: ast::expr_vec(descs, ast::imm, ast::sk_unique),
          span: ast::dummy_sp()};
}

fn mk_test_desc_rec(cx: &test_ctxt, test: test) -> @ast::expr {
    let path = test.path;

    log #fmt("encoding %s", ast::path_name_i(path));

    let name_lit: ast::lit =
        nospan(ast::lit_str(ast::path_name_i(path), ast::sk_rc));
    let name_expr: ast::expr =
        {id: cx.next_node_id(),
         node: ast::expr_lit(@name_lit),
         span: ast::dummy_sp()};

    let name_field: ast::field =
        nospan({mut: ast::imm, ident: "name", expr: @name_expr});

    let fn_path: ast::path =
        nospan({global: false, idents: path, types: ~[]});

    let fn_expr: ast::expr =
        {id: cx.next_node_id(),
         node: ast::expr_path(fn_path),
         span: ast::dummy_sp()};

    let fn_field: ast::field =
        nospan({mut: ast::imm, ident: "fn", expr: @fn_expr});

    let ignore_lit: ast::lit = nospan(ast::lit_bool(test.ignore));

    let ignore_expr: ast::expr =
        {id: cx.next_node_id(),
         node: ast::expr_lit(@ignore_lit),
         span: ast::dummy_sp()};

    let ignore_field: ast::field =
        nospan({mut: ast::imm, ident: "ignore", expr: @ignore_expr});

    let desc_rec_: ast::expr_ =
        ast::expr_rec(~[name_field, fn_field, ignore_field], option::none);
    let desc_rec: ast::expr =
        {id: cx.next_node_id(), node: desc_rec_, span: ast::dummy_sp()};
    ret @desc_rec;
}

fn mk_main(cx: &test_ctxt) -> @ast::item {

    let args_mt: ast::mt = {ty: @nospan(ast::ty_str), mut: ast::imm};
    let args_ty: ast::ty = nospan(ast::ty_vec(args_mt));

    let args_arg: ast::arg =
        {mode: ast::val, ty: @args_ty, ident: "args", id: cx.next_node_id()};

    let ret_ty = nospan(ast::ty_nil);

    let decl: ast::fn_decl =
        {inputs: ~[args_arg],
         output: @ret_ty,
         purity: ast::impure_fn,
         il: ast::il_normal,
         cf: ast::return,
         constraints: ~[]};
    let proto = ast::proto_fn;

    let test_main_call_expr = mk_test_main_call(cx);

    let body_: ast::blk_ =
        {stmts: ~[],
         expr: option::some(test_main_call_expr),
         id: cx.next_node_id()};
    let body = {node: body_, span: ast::dummy_sp()};

    let fn_ = {decl: decl, proto: proto, body: body};

    let item_ = ast::item_fn(fn_, ~[]);
    let item: ast::item =
        {ident: "main",
         attrs: ~[],
         id: cx.next_node_id(),
         node: item_,
         span: ast::dummy_sp()};
    ret @item;
}

fn mk_test_main_call(cx: &test_ctxt) -> @ast::expr {

    // Get the args passed to main so we can pass the to test_main
    let args_path: ast::path =
        nospan({global: false, idents: ~["args"], types: ~[]});

    let args_path_expr_: ast::expr_ = ast::expr_path(args_path);

    let args_path_expr: ast::expr =
        {id: cx.next_node_id(),
         node: args_path_expr_,
         span: ast::dummy_sp()};

    // Call __test::test to generate the vector of test_descs
    let test_path: ast::path =
        nospan({global: false, idents: ~["tests"], types: ~[]});

    let test_path_expr_: ast::expr_ = ast::expr_path(test_path);

    let test_path_expr: ast::expr =
        {id: cx.next_node_id(),
         node: test_path_expr_,
         span: ast::dummy_sp()};

    let test_call_expr_: ast::expr_ = ast::expr_call(@test_path_expr, ~[]);

    let test_call_expr: ast::expr =
        {id: cx.next_node_id(),
         node: test_call_expr_,
         span: ast::dummy_sp()};

    // Call std::test::test_main
    let test_main_path: ast::path =
        nospan({global: false,
                idents: ~["std", "test", "test_main"],
                types: ~[]});

    let test_main_path_expr_: ast::expr_ = ast::expr_path(test_main_path);

    let test_main_path_expr: ast::expr =
        {id: cx.next_node_id(),
         node: test_main_path_expr_,
         span: ast::dummy_sp()};

    let test_main_call_expr_: ast::expr_ =
        ast::expr_call(@test_main_path_expr,
                       ~[@args_path_expr, @test_call_expr]);

    let test_main_call_expr: ast::expr =
        {id: cx.next_node_id(),
         node: test_main_call_expr_,
         span: ast::dummy_sp()};

    ret @test_main_call_expr;
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
