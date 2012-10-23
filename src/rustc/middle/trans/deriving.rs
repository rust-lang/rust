// Translation of automatically-derived trait implementations. This handles
// enums and structs only; other types cannot be automatically derived.

use middle::trans::base::get_insn_ctxt;
use middle::trans::common::crate_ctxt;
use syntax::ast::{ident, node_id, ty_param};
use syntax::ast_map::path;

/// The main "translation" pass for automatically-derived impls. Generates
/// code for monomorphic methods only. Other methods will be generated when
/// they are invoked with specific type parameters; see
/// `trans::base::lval_static_fn()` or `trans::base::monomorphic_fn()`.
pub fn trans_deriving_impl(ccx: @crate_ctxt, _path: path, _name: ident,
                           tps: ~[ty_param], _self_ty: Option<ty::t>,
                           _id: node_id) {
    let _icx = ccx.insn_ctxt("deriving::trans_deriving_impl");
    if tps.len() > 0 { return; }

    // XXX: Unimplemented.
}

