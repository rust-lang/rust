// A pass that checks to make sure private fields and methods aren't used
// outside their scopes.

use /*mod*/ syntax::ast;
use /*mod*/ syntax::visit;
use syntax::ast::{expr_field, ident, item_class, local_crate, node_id};
use syntax::ast::{private};
use ty::ty_class;

use core::util::ignore;
use dvec::DVec;
use send_map::linear::LinearMap;

fn check_crate(tcx: ty::ctxt, crate: @ast::crate) {
    let privileged_structs = @DVec();

    let add_privileged_structs = |items: &[@ast::item]| {
        let mut count = 0;
        for items.each |item| {
            match item.node {
                item_class(*) => {
                    privileged_structs.push(item.id);
                    count += 1;
                }
                _ => {}
            }
        }
        count
    };

    let visitor = visit::mk_vt(@{
        visit_mod: |the_module, span, node_id, env, visitor| {
            let n_added = add_privileged_structs(the_module.items);

            visit::visit_mod(the_module, span, node_id, env, visitor);

            for n_added.times {
                ignore(privileged_structs.pop());
            }
        },
        visit_expr: |expr, env, visitor| {
            match expr.node {
                expr_field(base, ident, _) => {
                    match ty::get(ty::expr_ty(tcx, base)).struct {
                        ty_class(id, _)
                        if id.crate != local_crate ||
                           !privileged_structs.contains(id.node) => {
                            let fields = ty::lookup_class_fields(tcx, id);
                            for fields.each |field| {
                                if field.ident != ident { again; }
                                if field.vis == private {
                                    tcx.sess.span_err(expr.span,
                                                      fmt!("field `%s` is \
                                                            private",
                                                           *tcx.sess
                                                               .parse_sess
                                                               .interner
                                                               .get(ident)));
                                }
                                break;
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }

            visit::visit_expr(expr, env, visitor);
        }
        with *visit::default_visitor()
    });
    visit::visit_crate(*crate, (), visitor);
}

