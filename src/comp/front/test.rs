// Code that generates a test runner to run all the tests in a crate

import std::option;
import std::ivec;
import syntax::ast;
import syntax::fold;
import syntax::print::pprust;
import front::attr;

export modify_for_testing;

type node_id_gen = @fn() -> ast::node_id;

type test_ctxt = @rec(node_id_gen next_node_id,
                      mutable ast::ident[] path,
                      mutable ast::ident[][] testfns);

// Traverse the crate, collecting all the test functions, eliding any
// existing main functions, and synthesizing a main test harness
fn modify_for_testing(@ast::crate crate) -> @ast::crate {

    // FIXME: This hackasaurus assumes that 200000 is a safe number to start
    // generating node_ids at (which is totally not the case). pauls is going
    // to land a patch that puts parse_sess into session, which will give us
    // access to the real next node_id.
    auto next_node_id = @mutable 200000;
    auto next_node_id_fn = @bind fn(@mutable ast::node_id next_node_id)
        -> ast::node_id {
        auto this_node_id = *next_node_id;
        *next_node_id = next_node_id + 1;
        ret this_node_id;
    } (next_node_id);

    let test_ctxt cx = @rec(next_node_id = next_node_id_fn,
                            mutable path = ~[],
                            mutable testfns = ~[]);

    auto precursor = rec(fold_crate = bind fold_crate(cx, _, _),
                         fold_item = bind fold_item(cx, _, _),
                         fold_mod = bind fold_mod(cx, _, _)
                         with *fold::default_ast_fold());

    auto fold = fold::make_fold(precursor);
    auto res = @fold.fold_crate(*crate);
    // FIXME: This is necessary to break a circular reference
    fold::dummy_out(fold);
    ret res;
}

fn fold_mod(&test_ctxt cx, &ast::_mod m,
            fold::ast_fold fld) -> ast::_mod {

    // Remove any defined main function from the AST so it doesn't clash with
    // the one we're going to add.  FIXME: This is sloppy. Instead we should
    // have some mechanism to indicate to the translation pass which function
    // we want to be main.
    fn nomain(&@ast::item item) -> option::t[@ast::item] {
        alt (item.node) {
            ast::item_fn(?f, _) {
                if (item.ident == "main") { option::none }
                else { option::some(item) }
            }
            _ { option::some(item) }
        }
    }

    auto mod_nomain = rec(view_items=m.view_items,
                          items=ivec::filter_map(nomain, m.items));
    ret fold::noop_fold_mod(mod_nomain, fld);
}

fn fold_crate(&test_ctxt cx, &ast::crate_ c,
              fold::ast_fold fld) -> ast::crate_ {
    auto folded = fold::noop_fold_crate(c, fld);

    // Add a special __test module to the crate that will contain code
    // generated for the test harness
    ret rec(module = add_test_module(cx, folded.module)
            with folded);
}


fn fold_item(&test_ctxt cx, &@ast::item i,
             fold::ast_fold fld) -> @ast::item {

    cx.path += ~[i.ident];
    log #fmt("current path: %s", ast::path_name_i(cx.path));

    if (is_test_fn(i)) {
        log "this is a test function";
        cx.testfns += ~[cx.path];
        log #fmt("have %u test functions", ivec::len(cx.testfns));
    }

    auto res = fold::noop_fold_item(i, fld);
    ivec::pop(cx.path);
    ret res;
}

fn is_test_fn(&@ast::item i) -> bool {
    auto has_test_attr = 
        ivec::len(attr::find_attrs_by_name(i.attrs, "test")) > 0u;

    fn has_test_signature(&@ast::item i) -> bool {
        alt (i.node) {
            case (ast::item_fn(?f, ?tps)) {
                auto input_cnt = ivec::len(f.decl.inputs);
                auto no_output = f.decl.output.node == ast::ty_nil;
                auto tparm_cnt = ivec::len(tps);
                input_cnt == 0u && no_output && tparm_cnt == 0u
            }
            case (_) { false }
        }
    }

    ret has_test_attr && has_test_signature(i);
}

fn add_test_module(&test_ctxt cx, &ast::_mod m) -> ast::_mod {
    auto testmod = mk_test_module(cx);
    ret rec(items=m.items + ~[testmod] with m);
}

/*

We're going to be building a module that looks more or less like:

mod __test {

  fn main(vec[str] args) -> int {
    std::test::test_main(args, tests())
  }

  fn tests() -> std::test::test_desc[] {
    ... the list of tests in the crate ...
  }
}

*/

fn mk_test_module(&test_ctxt cx) -> @ast::item {
    // A function that generates a vector of test descriptors to feed to the
    // test runner
    auto testsfn = mk_tests(cx);
    // The synthesized main function which will call the console test runner
    // with our list of tests
    auto mainfn = mk_main(cx);
    let ast::_mod testmod = rec(view_items=~[],
                                items=~[mainfn, testsfn]);
    auto item_ = ast::item_mod(testmod);
    let ast::item item = rec(ident = "__test",
                              attrs = ~[],
                              id = cx.next_node_id(),
                              node = item_,
                              span = rec(lo=0u, hi=0u));

    log #fmt("Synthetic test module:\n%s\n", pprust::item_to_str(@item));

    ret @item;
}

fn nospan[T](&T t) -> ast::spanned[T] {
    ret rec(node=t,
            span=rec(lo=0u,hi=0u));
}

fn mk_tests(&test_ctxt cx) -> @ast::item {
    auto ret_ty = mk_test_desc_ivec_ty(cx);

    let ast::fn_decl decl = rec(inputs = ~[],
                                output = ret_ty,
                                purity = ast::impure_fn,
                                cf = ast::return,
                                constraints = ~[]);
    auto proto = ast::proto_fn;
    
    // The vector of test_descs for this crate
    auto test_descs = mk_test_desc_vec(cx);

    let ast::block_ body_= rec(stmts = ~[],
                               expr = option::some(test_descs),
                               id = cx.next_node_id());
    auto body = nospan(body_);

    auto fn_ = rec(decl = decl,
                   proto = proto,
                   body = body);

    auto item_ = ast::item_fn(fn_, ~[]);
    let ast::item item = rec(ident = "tests",
                             attrs = ~[],
                             id = cx.next_node_id(),
                             node = item_,
                             span = rec(lo=0u, hi=0u));
    ret @item;
}

fn empty_fn_ty() -> ast::ty {
    auto proto = ast::proto_fn;
    auto input_ty = ~[];
    auto ret_ty = @nospan(ast::ty_nil);
    auto cf = ast::return;
    auto constrs = ~[];
    ret nospan(ast::ty_fn(proto, input_ty, ret_ty, cf, constrs));
}

// The ast::ty of std::test::test_desc[]
fn mk_test_desc_ivec_ty(&test_ctxt cx) -> @ast::ty {
    let ast::path test_desc_ty_path = nospan(rec(global = false,
                                                 idents = ~["std",
                                                            "test",
                                                            "test_desc"],
                                                 types = ~[]));

    let ast::ty test_desc_ty = nospan(ast::ty_path(test_desc_ty_path,
                                                   cx.next_node_id()));

    let ast::mt ivec_mt = rec(ty = @test_desc_ty,
                              mut = ast::imm);

    ret @nospan(ast::ty_ivec(ivec_mt));
}

fn mk_test_desc_vec(&test_ctxt cx) -> @ast::expr {
    log #fmt("building test vector from %u tests",
             ivec::len(cx.testfns));
    auto descs = ~[];
    for (ast::ident[] testpath in cx.testfns) {
        log #fmt("encoding %s", ast::path_name_i(testpath));
        auto path = testpath;
        descs += ~[mk_test_desc_rec(cx, path)];
    }

    ret @rec(id = cx.next_node_id(),
             node = ast::expr_vec(descs, ast::imm, ast::sk_unique),
             span = rec(lo=0u,hi=0u));
}

fn mk_test_desc_rec(&test_ctxt cx, ast::ident[] path) -> @ast::expr {

    let ast::lit name_lit = nospan(ast::lit_str(ast::path_name_i(path),
                                                ast::sk_rc));
    let ast::expr name_expr = rec(id = cx.next_node_id(),
                                  node = ast::expr_lit(@name_lit),
                                  span = rec(lo=0u, hi=0u));

    let ast::field name_field = nospan(rec(mut = ast::imm,
                                           ident = "name",
                                           expr = @name_expr));

    let ast::path fn_path = nospan(rec(global = false,
                                       idents = path,
                                       types = ~[]));

    let ast::expr fn_expr = rec(id = cx.next_node_id(),
                                node = ast::expr_path(fn_path),
                                span = rec(lo=0u, hi=0u));

    let ast::field fn_field = nospan(rec(mut = ast::imm,
                                         ident = "fn",
                                         expr = @fn_expr));

    let ast::expr_ desc_rec_ = ast::expr_rec(~[name_field, fn_field],
                                             option::none);
    let ast::expr desc_rec = rec(id = cx.next_node_id(),
                                 node = desc_rec_,
                                 span = rec(lo=0u, hi=0u));
    ret @desc_rec;
}

fn mk_main(&test_ctxt cx) -> @ast::item {

    let ast::mt args_mt = rec(ty = @nospan(ast::ty_str),
                              mut = ast::imm);
    let ast::ty args_ty = nospan(ast::ty_vec(args_mt));

    let ast::arg args_arg = rec(mode = ast::val,
                                ty = @args_ty,
                                ident = "args",
                                id = cx.next_node_id());

    auto ret_ty = nospan(ast::ty_int);

    let ast::fn_decl decl = rec(inputs = ~[args_arg],
                                output = @ret_ty,
                                purity = ast::impure_fn,
                                cf = ast::return,
                                constraints = ~[]);
    auto proto = ast::proto_fn;

    auto test_main_call_expr = mk_test_main_call(cx);

    let ast::block_ body_ = rec(stmts = ~[],
                                expr = option::some(test_main_call_expr),
                                id = cx.next_node_id());
    auto body = rec(node = body_, span = rec(lo=0u, hi=0u));

    auto fn_ = rec(decl = decl,
                   proto = proto,
                   body = body);

    auto item_ = ast::item_fn(fn_, ~[]);
    let ast::item item = rec(ident = "main",
                             attrs = ~[],
                             id = cx.next_node_id(),
                             node = item_,
                             span = rec(lo=0u, hi=0u));
    ret @item;
}

fn mk_test_main_call(&test_ctxt cx) -> @ast::expr {

    // Get the args passed to main so we can pass the to test_main
    let ast::path args_path = nospan(rec(global = false,
                                         idents = ~["args"],
                                         types = ~[]));

    let ast::expr_ args_path_expr_ = ast::expr_path(args_path);

    let ast::expr args_path_expr = rec(id = cx.next_node_id(),
                                       node = args_path_expr_,
                                       span = rec(lo=0u, hi=0u));

    // Call __test::test to generate the vector of test_descs
    let ast::path test_path = nospan(rec(global = false,
                                         idents = ~["tests"],
                                         types = ~[]));

    let ast::expr_ test_path_expr_ = ast::expr_path(test_path);

    let ast::expr test_path_expr = rec(id = cx.next_node_id(),
                                       node = test_path_expr_,
                                       span = rec(lo=0u, hi=0u));

    let ast::expr_ test_call_expr_ = ast::expr_call(@test_path_expr, ~[]);

    let ast::expr test_call_expr = rec(id = cx.next_node_id(),
                                       node = test_call_expr_,
                                       span = rec(lo=0u, hi=0u));

    // Call std::test::test_main
    let ast::path test_main_path = nospan(rec(global = false,
                                              idents = ~["std",
                                                         "test",
                                                         "test_main"],
                                              types = ~[]));

    let ast::expr_ test_main_path_expr_
        = ast::expr_path(test_main_path);

    let ast::expr test_main_path_expr = rec(id = cx.next_node_id(),
                                            node = test_main_path_expr_,
                                            span = rec(lo=0u, hi=0u));

    let ast::expr_ test_main_call_expr_ 
        = ast::expr_call(@test_main_path_expr, ~[@args_path_expr,
                                                 @test_call_expr]);

    let ast::expr test_main_call_expr = rec(id = cx.next_node_id(),
                                            node = test_main_call_expr_,
                                            span = rec(lo=0u, hi=0u));

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
