// Code that generates a test runner to run all the tests in a crate

import std::option;
import syntax::ast;
import syntax::fold;

export modify_for_testing;

type node_id_gen = @fn() -> ast::node_id;

type test_ctxt = rec(node_id_gen next_node_id);

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

    auto cx = rec(next_node_id = next_node_id_fn);

    auto precursor = rec(fold_crate = bind fold_crate(cx, _, _)
                         with *fold::default_ast_fold());

    auto fold = fold::make_fold(precursor);
    auto res = @fold.fold_crate(*crate);
    // FIXME: This is necessary to break a circular reference
    fold::dummy_out(fold);
    ret res;
}

fn fold_crate(&test_ctxt cx, &ast::crate_ c,
              fold::ast_fold fld) -> ast::crate_ {
    auto folded = fold::noop_fold_crate(c, fld);

    // Add a special __test module to the crate that will contain code
    // generated for the test harness
    ret rec(module = add_test_module(cx, folded.module)
            with folded);
}

fn add_test_module(&test_ctxt cx, &ast::_mod m) -> ast::_mod {
    auto testmod = mk_test_module(cx);
    ret rec(items=m.items + ~[testmod] with m);
}

fn mk_test_module(&test_ctxt cx) -> @ast::item {
    auto mainfn = mk_main(cx);
    let ast::_mod testmod = rec(view_items=~[], items=~[mainfn]);
    auto item_ = ast::item_mod(testmod);
    let ast::item item = rec(ident = "__test",
                             attrs = ~[],
                             id = cx.next_node_id(),
                             node = item_,
                             span = rec(lo=0u, hi=0u));
    ret @item;
}

fn mk_main(&test_ctxt cx) -> @ast::item {
    auto ret_ty = @rec(node=ast::ty_nil,
                       span=rec(lo=0u, hi=0u));

    let ast::fn_decl decl = rec(inputs = ~[],
                                output = ret_ty,
                                purity = ast::impure_fn,
                                cf = ast::return,
                                constraints = ~[]);
    auto proto = ast::proto_fn;

    let ast::block_ body_ = rec(stmts = ~[],
                                 expr = option::none,
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

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
