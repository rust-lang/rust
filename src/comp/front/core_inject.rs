import driver::session::session;
import syntax::ast;
import syntax::codemap;

export maybe_inject_libcore_ref;

fn maybe_inject_libcore_ref(sess: session,
                            crate: @ast::crate) -> @ast::crate {
    if sess.opts.libcore {
        inject_libcore_ref(sess, crate)
    } else {
        crate
    }
}

fn inject_libcore_ref(sess: session,
                      crate: @ast::crate) -> @ast::crate {

    fn spanned<T: copy>(x: T) -> @ast::spanned<T> {
        ret @{node: x,
              span: {lo: 0u, hi: 0u,
                     expanded_from: codemap::os_none}};
    }

    let n1 = sess.next_node_id();
    let n2 = sess.next_node_id();

    let vi1 = spanned(ast::view_item_use("core", [], n1));
    let vi2 = spanned(ast::view_item_import_glob(@["core"], n2));

    let vis = [vi1, vi2] + crate.node.module.view_items;

    ret @{node: {module: { view_items: vis with crate.node.module }
                 with crate.node} with *crate }
}
