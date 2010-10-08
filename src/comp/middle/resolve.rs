import front.ast;
import driver.session;
import util.common.span;

type env = ();

fn resolve_name(&env e, &span sp, ast.name_ n) -> ast.name {
    auto s = "resolving name: ";
    s += n.ident;
    log s;
    ret fold.respan[ast.name_](sp, n);
}

fn resolve_crate(session.session sess, @ast.crate crate) -> @ast.crate {
    let fold.ast_fold[env] fld = fold.new_identity_fold[env]();
    fld = @rec( fold_name = bind resolve_name(_,_,_)
                with *fld );
    ret fold.fold_crate[env]((), fld, crate);
}

// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
