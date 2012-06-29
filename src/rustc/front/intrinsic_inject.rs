import driver::session::session;
import syntax::parse;
import syntax::ast;

export inject_intrinsic;

fn inject_intrinsic(sess: session,
                    crate: @ast::crate) -> @ast::crate {

    let intrinsic_module = @#include_str("intrinsic.rs");

    let item = parse::parse_item_from_source_str("<intrinsic>",
                                                 intrinsic_module,
                                                 sess.opts.cfg,
                                                 []/~, ast::public,
                                                 sess.parse_sess);
    let item =
        alt item {
          some(i) { i }
          none {
            sess.fatal("no item found in intrinsic module");
          }
        };

    let items = vec::append([item]/~, crate.node.module.items);

    ret @{node: {module: { items: items with crate.node.module }
                 with crate.node} with *crate }
}
