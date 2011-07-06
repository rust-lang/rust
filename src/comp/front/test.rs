import driver::session;
import syntax::ast;
import syntax::fold;

export modify_for_testing;

type test_ctxt = rec(@session::session sess);

// Traverse the crate, collecting all the test functions, eliding any
// existing main functions, and synthesizing a main test harness
fn modify_for_testing(&session::session sess,
                      @ast::crate crate) -> @ast::crate {

  auto cx = rec(sess = @sess);

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
  ret rec(module = add_test_module(folded.module)
          with folded);
}

fn add_test_module(&ast::_mod m) -> ast::_mod {
  ret m;
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
