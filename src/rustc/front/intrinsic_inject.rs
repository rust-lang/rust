import driver::session::session;
import syntax::parse;
import syntax::ast;

export inject_intrinsic;

fn inject_intrinsic(sess: session,
                    crate: @ast::crate) -> @ast::crate {

    let intrinsic_module = @include_str!{"intrinsic.rs"};

    let item = parse::parse_item_from_source_str(~"<intrinsic>",
                                                 intrinsic_module,
                                                 sess.opts.cfg,
                                                 ~[],
                                                 sess.parse_sess);
    let item =
        match item {
          some(i) => i,
          none => {
            sess.fatal(~"no item found in intrinsic module");
          }
        };

    let items = vec::append(~[item], crate.node.module.items);

    return @{node: {module: { items: items with crate.node.module }
                 with crate.node} with *crate }
}
