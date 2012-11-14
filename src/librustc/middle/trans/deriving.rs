// Translation of automatically-derived trait implementations. This handles
// enums and structs only; other types cannot be automatically derived.

use lib::llvm::llvm::{LLVMCountParams, LLVMGetParam};
use middle::trans::base::{GEP_enum, finish_fn, get_insn_ctxt, get_item_val};
use middle::trans::base::{new_fn_ctxt, sub_block, top_scope_block};
use middle::trans::build::{AddCase, Br, CondBr, GEPi, Load, PointerCast};
use middle::trans::build::{Store, Switch, Unreachable, ValueRef};
use middle::trans::callee;
use middle::trans::callee::{ArgVals, Callee, DontAutorefArg, Method};
use middle::trans::callee::{MethodData};
use middle::trans::common;
use middle::trans::common::{C_bool, C_int, T_ptr, block, crate_ctxt};
use middle::trans::common::{fn_ctxt};
use middle::trans::expr::SaveIn;
use middle::trans::type_of::type_of;
use middle::ty::DerivedFieldInfo;
use middle::typeck::method_static;
use syntax::ast;
use syntax::ast::{def_id, ident, node_id, ty_param};
use syntax::ast_map::path;
use syntax::ast_util;
use syntax::ast_util::local_def;

use core::dvec::DVec;
use core::dvec;
use core::libc::c_uint;

/// The kind of deriving method this is.
enum DerivingKind {
    BoolKind,   // fn f(&self, other: &other) -> bool
    UnitKind,   // fn f(&self) -> ()
}

impl DerivingKind {
    static fn of_item(ccx: @crate_ctxt, method_did: ast::def_id)
                   -> DerivingKind {
        let item_type = ty::lookup_item_type(ccx.tcx, method_did).ty;
        match ty::get(item_type).sty {
            ty::ty_fn(ref f) => {
                match ty::get(f.sig.output).sty {
                    ty::ty_bool => BoolKind,
                    ty::ty_nil => UnitKind,
                    _ => {
                        // FIXME (#3957): Report this earlier.
                        ccx.tcx.sess.fatal(~"attempt to automatically derive \
                                             derive an implementation of a \
                                             function returning something \
                                             other than bool or ()");
                    }
                }
            }
            _ => {
                ccx.tcx.sess.bug(~"DerivingKind::of_item(): method def ID \
                                   didn't have a function type");
            }
        }
    }
}

/// The main "translation" pass for the automatically-derived methods in
/// an impl. Generates code for monomorphic methods only. Other methods will
/// be generated when they are invoked with specific type parameters; see
/// `trans::base::lval_static_fn()` or `trans::base::monomorphic_fn()`.
pub fn trans_deriving_impl(ccx: @crate_ctxt,
                           _path: path,
                           _name: ident,
                           tps: ~[ty_param],
                           id: node_id) {
    let _icx = ccx.insn_ctxt("deriving::trans_deriving_impl");
    if tps.len() > 0 { return; }

    let impl_def_id = local_def(id);
    let self_ty = ty::lookup_item_type(ccx.tcx, impl_def_id);

    match ccx.tcx.automatically_derived_methods_for_impl.find(impl_def_id) {
        Some(copy method_dids) => {
            for method_dids.each |method_did| {
                let kind = DerivingKind::of_item(ccx, *method_did);
                let llfn = get_item_val(ccx, method_did.node);
                match ty::get(self_ty.ty).sty {
                    ty::ty_class(*) => {
                        trans_deriving_struct_method(ccx, llfn, impl_def_id,
                                                     self_ty.ty, kind);
                    }
                    ty::ty_enum(*) => {
                        trans_deriving_enum_method(ccx, llfn, impl_def_id,
                                                   self_ty.ty, kind);
                    }
                    _ => {
                        ccx.tcx.sess.bug(~"translation of non-struct \
                                           deriving method");
                    }
                }
            }
        }
        None => {}  // Nothing to do.
    }
}

fn get_extra_params(llfn: ValueRef, kind: DerivingKind) -> ~[ValueRef] {
    let n_params = LLVMCountParams(llfn) as uint;

    let initial_extra_param;
    match kind {
        BoolKind => initial_extra_param = 3,
        UnitKind => initial_extra_param = 2,
    }

    let extra_params = DVec();
    for uint::range(initial_extra_param, n_params) |i| {
        extra_params.push(LLVMGetParam(llfn, i as c_uint));
    }

    return dvec::unwrap(move extra_params);
}

fn trans_deriving_struct_method(ccx: @crate_ctxt,
                                llfn: ValueRef,
                                impl_did: def_id,
                                self_ty: ty::t,
                                kind: DerivingKind) {
    let _icx = ccx.insn_ctxt("trans_deriving_struct_method");
    let fcx = new_fn_ctxt(ccx, ~[], llfn, None);
    let top_bcx = top_scope_block(fcx, None);
    let lltop = top_bcx.llbb;
    let mut bcx = top_bcx;

    let llextraparams = get_extra_params(llfn, kind);

    let llselfty = type_of(ccx, self_ty);
    let llselfval = PointerCast(bcx, fcx.llenv, T_ptr(llselfty));

    // If there is an "other" value, then get it. The "other" value is the
    // value we're comparing against in the case of Eq and Ord.
    let llotherval_opt;
    match kind {
        BoolKind => llotherval_opt = Some(LLVMGetParam(llfn, 2)),
        UnitKind => llotherval_opt = None
    }

    let struct_field_tys;
    match ty::get(self_ty).sty {
        ty::ty_class(struct_id, ref struct_substs) => {
            struct_field_tys = ty::class_items_as_fields(
                ccx.tcx, struct_id, struct_substs);
        }
        _ => {
            ccx.tcx.sess.bug(~"passed non-struct to \
                               trans_deriving_struct_method");
        }
    }

    // Iterate over every element of the struct.
    for ccx.tcx.deriving_struct_methods.get(impl_did).eachi
            |i, derived_method_info| {
        let llselfval = GEPi(bcx, llselfval, [0, 0, i]);

        let llotherval_opt = llotherval_opt.map(
            |llotherval| GEPi(bcx, *llotherval, [0, 0, i]));

        let self_ty = struct_field_tys[i].mt.ty;
        bcx = call_substructure_method(bcx,
                                       derived_method_info,
                                       self_ty,
                                       llselfval,
                                       llotherval_opt,
                                       llextraparams);

        // If this derived method is of boolean kind, return immediately if
        // the call to the substructure method returned false.
        match kind {
            BoolKind => {
                let next_block = sub_block(top_bcx, ~"next");
                let llcond = Load(bcx, fcx.llretptr);
                CondBr(bcx, llcond, next_block.llbb, fcx.llreturn);
                bcx = next_block;
            }
            UnitKind => {}  // Unconditionally continue.
        }
    }

    // Store true if necessary.
    match kind {
        BoolKind => Store(bcx, C_bool(true), fcx.llretptr),
        UnitKind => {}
    }

    Br(bcx, fcx.llreturn);

    finish_fn(fcx, lltop);
}

// This could have been combined with trans_deriving_struct_method, but it
// would probably be too big and hard to understand.
fn trans_deriving_enum_method(ccx: @crate_ctxt,
                              llfn: ValueRef,
                              impl_did: def_id,
                              self_ty: ty::t,
                              kind: DerivingKind) {
    let _icx = ccx.insn_ctxt("trans_deriving_enum_method");
    let fcx = new_fn_ctxt(ccx, ~[], llfn, None);
    let top_bcx = top_scope_block(fcx, None);
    let lltop = top_bcx.llbb;
    let mut bcx = top_bcx;

    let llextraparams = get_extra_params(llfn, kind);

    let llselfty = type_of(ccx, self_ty);
    let llselfval = PointerCast(bcx, fcx.llenv, T_ptr(llselfty));

    let llotherval_opt;
    match kind {
        UnitKind => llotherval_opt = None,
        BoolKind => llotherval_opt = Some(LLVMGetParam(llfn, 2))
    }

    let enum_id, enum_substs, enum_variant_infos;
    match ty::get(self_ty).sty {
        ty::ty_enum(found_enum_id, ref found_enum_substs) => {
            enum_id = found_enum_id;
            enum_substs = copy *found_enum_substs;
            enum_variant_infos = ty::substd_enum_variants(
                ccx.tcx, enum_id, &enum_substs);
        }
        _ => {
            ccx.tcx.sess.bug(~"passed non-enum to \
                               trans_deriving_enum_method");
        }
    }

    // Create the "no match" basic block, if necessary. This is a basic block
    // that does nothing more than return false.
    let nomatch_bcx_opt;
    match kind {
        BoolKind => {
            let nomatch_bcx = sub_block(top_bcx, ~"no_match");
            Store(nomatch_bcx, C_bool(false), fcx.llretptr);
            Br(nomatch_bcx, fcx.llreturn);
            nomatch_bcx_opt = Some(nomatch_bcx);
        }
        UnitKind => nomatch_bcx_opt = None
    }

    // Create the "unreachable" basic block.
    let unreachable_bcx = sub_block(top_bcx, ~"unreachable");
    Unreachable(unreachable_bcx);

    // Get the deriving enum method info.
    let deriving_enum_methods = ccx.tcx.deriving_enum_methods.get(impl_did);
    let n_variants = deriving_enum_methods.len();

    if n_variants != 1 {
        // Grab the two discriminants.
        let llselfdiscrim = Load(bcx, GEPi(bcx, llselfval, [0, 0]));
        let llotherdiscrim_opt = llotherval_opt.map(
            |llotherval| Load(bcx, GEPi(bcx, *llotherval, [0, 0])));

        // Skip over the discriminants and compute the address of the payload.
        let llselfpayload = GEPi(bcx, llselfval, [0, 1]);
        let llotherpayload_opt = llotherval_opt.map(
            |llotherval| GEPi(bcx, *llotherval, [0, 1]));

        // Create basic blocks for the outer switch.
        let outer_bcxs = vec::from_fn(
            deriving_enum_methods.len(),
            |i| sub_block(top_bcx, fmt!("outer_%u", i)));

        // For each basic block in the outer switch...
        for outer_bcxs.eachi |self_variant_index, bcx| {
            // Create the matching basic block for the inner switch.
            let top_match_bcx = sub_block(top_bcx, fmt!("maybe_match_%u",
                                                        self_variant_index));
            let mut match_bcx = top_match_bcx;

            // Compare each variant.
            for deriving_enum_methods[self_variant_index].eachi
                    |i, derived_method_info| {
                let variant_def_id =
                        enum_variant_infos[self_variant_index].id;
                let llselfval = GEP_enum(match_bcx, llselfpayload, enum_id,
                                         variant_def_id, enum_substs.tps, i);

                let llotherval_opt = llotherpayload_opt.map(|llotherpayload|
                    GEP_enum(match_bcx, *llotherpayload, enum_id,
                             variant_def_id, enum_substs.tps, i));

                let self_ty = enum_variant_infos[self_variant_index].args[i];
                match_bcx = call_substructure_method(match_bcx,
                                                     derived_method_info,
                                                     self_ty,
                                                     llselfval,
                                                     llotherval_opt,
                                                     llextraparams);

                // If this is a boolean-kind deriving method, then return
                // immediately if the call to the substructure returned false.
                match kind {
                    BoolKind => {
                        let next_bcx = sub_block(top_bcx,
                                                 fmt!("next_%u_%u",
                                                      self_variant_index,
                                                      i));
                        let llcond = Load(match_bcx, fcx.llretptr);
                        CondBr(match_bcx,
                               llcond,
                               next_bcx.llbb,
                               fcx.llreturn);
                        match_bcx = next_bcx;
                    }
                    UnitKind => {}
                }
            }

            // Store true in the return pointer if this is a boolean-kind
            // deriving method.
            match kind {
                BoolKind => Store(match_bcx, C_bool(true), fcx.llretptr),
                UnitKind => {}
            }

            // Finish up the matching block.
            Br(match_bcx, fcx.llreturn);

            // If this is a boolean-kind derived method, build the inner
            // switch. Otherwise, just jump to the matching case.
            match llotherdiscrim_opt {
                None => Br(*bcx, top_match_bcx.llbb),
                Some(copy llotherdiscrim) => {
                    let llswitch = Switch(*bcx,
                                          llotherdiscrim,
                                          unreachable_bcx.llbb,
                                          n_variants);
                    for uint::range(0, n_variants) |other_variant_index| {
                        let discriminant =
                            enum_variant_infos[other_variant_index].disr_val;
                        if self_variant_index == other_variant_index {
                            // This is the potentially-matching case.
                            AddCase(llswitch,
                                    C_int(ccx, discriminant),
                                    top_match_bcx.llbb);
                        } else {
                            // This is always a non-matching case.
                            AddCase(llswitch,
                                    C_int(ccx, discriminant),
                                    nomatch_bcx_opt.get().llbb);
                        }
                    }
                }
            }
        }

        // Now build the outer switch.
        let llswitch = Switch(top_bcx, llselfdiscrim, unreachable_bcx.llbb,
                              n_variants);
        for outer_bcxs.eachi |self_variant_index, outer_bcx| {
            let discriminant =
                enum_variant_infos[self_variant_index].disr_val;
            AddCase(llswitch, C_int(ccx, discriminant), outer_bcx.llbb);
        }
    } else {
        ccx.tcx.sess.unimpl(~"degenerate enum deriving");
    }

    // Finish up the function.
    finish_fn(fcx, lltop);
}

fn call_substructure_method(bcx: block,
                            derived_field_info: &DerivedFieldInfo,
                            self_ty: ty::t,
                            llselfval: ValueRef,
                            llotherval_opt: Option<ValueRef>,
                            llextraparams: &[ValueRef]) -> block {
    let fcx = bcx.fcx;
    let ccx = fcx.ccx;

    let target_method_def_id;
    match derived_field_info.method_origin {
        method_static(did) => target_method_def_id = did,
        _ => fail ~"derived method didn't resolve to a static method"
    }

    let fn_expr_tpbt = ty::lookup_item_type(ccx.tcx, target_method_def_id);
    debug!("(calling substructure method) substructure method has %u \
            parameter(s), vtable result is %?",
           fn_expr_tpbt.bounds.len(),
           derived_field_info.vtable_result);

    // Get the substructure method we need to call. This may involve
    // code generation in the case of generics, default methods, or cross-
    // crate inlining.
    let fn_data = callee::trans_fn_ref_with_vtables(bcx,
                                                    target_method_def_id,
                                                    0,     // ref id
                                                    *derived_field_info.
                                                 type_parameter_substitutions,
                                                    derived_field_info.
                                                        vtable_result);
    let llfn = fn_data.llfn;

    // Create the callee.
    let cb: &fn(block) -> Callee = |bloc| {
        Callee {
            bcx: bloc,
            data: Method(MethodData {
                llfn: llfn,
                llself: llselfval,
                self_ty: self_ty,
                self_mode: ast::by_copy
            })
        }
    };

    // Build up the argument list.
    let llargvals = DVec();
    for llotherval_opt.each |llotherval| { llargvals.push(*llotherval); }
    for llextraparams.each |llextraparam| { llargvals.push(*llextraparam); }

    // And perform the call.
    callee::trans_call_inner(bcx,
                             None,
                             fn_expr_tpbt.ty,
                             ty::mk_bool(ccx.tcx),
                             cb,
                             ArgVals(dvec::unwrap(move llargvals)),
                             SaveIn(fcx.llretptr),
                             DontAutorefArg)
}

