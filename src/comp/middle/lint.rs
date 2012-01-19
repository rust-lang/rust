import driver::session::session;
import middle::ty::ctxt;
import syntax::{ast, visit};

type crate_ctxt = {tcx: ty::ctxt};

enum option {
    ctypes;
}

fn check_crate(tcx: ty::ctxt, crate: @ast::crate,
               checks: [option], timer: block(str, fn@())) {
    let ccx = @{tcx: tcx};
    vec::iter(checks) {|c|
        alt c {
          ctypes {
            timer("ctypes usage checking", bind check_ctypes(ccx, crate))
          }
        }
    }
}

fn check_ctypes(ccx: @crate_ctxt, crate: @ast::crate) {
    fn check_native_fn(ccx: @crate_ctxt, decl: ast::fn_decl) {
        let tys = vec::map(decl.inputs) {|a| a.ty };
        for ty in (tys + [decl.output]) {
            alt ty.node {
              ast::ty_int(ast::ty_i) {
                ccx.tcx.sess.span_warn(
                    ty.span,
                    "found rust type `int` in native module, while \
                     ctypes::c_int or ctypes::long should be used");
              }
              ast::ty_uint(ast::ty_u) {
                ccx.tcx.sess.span_warn(
                    ty.span,
                    "found rust type `uint` in native module, while \
                     ctypes::c_uint or ctypes::ulong should be used");
              }
              _ { }
            }
        }
    }

    fn check_item(ccx: @crate_ctxt, it: @ast::item) {
        alt it.node {
          ast::item_native_mod(nmod) {
            for ni in nmod.items {
                alt ni.node {
                  ast::native_item_fn(decl, tps) {
                    check_native_fn(ccx, decl);
                  }
                  _ { }
                }
            }
          }
          _ {/* nothing to do */ }
        }
    }

    let visit = visit::mk_simple_visitor(@{
        visit_item: bind check_item(ccx, _)
        with *visit::default_simple_visitor()
    });
    visit::visit_crate(*crate, (), visit);
}
//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//
