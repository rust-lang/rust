// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::rc::Rc;

use arena::TypedArena;
use back::link;
use llvm::{ValueRef, get_params};
use middle::def_id::DefId;
use middle::infer;
use middle::subst::{Subst, Substs};
use middle::subst;
use middle::traits;
use trans::base::*;
use trans::build::*;
use trans::callee::{Callee, Virtual, ArgVals,
                    trans_fn_pointer_shim, trans_fn_ref_with_substs};
use trans::closure;
use trans::common::*;
use trans::consts;
use trans::datum::*;
use trans::debuginfo::DebugLoc;
use trans::declare;
use trans::expr;
use trans::glue;
use trans::machine;
use trans::type_::Type;
use trans::type_of::*;
use middle::ty::{self, Ty, TyCtxt, TypeFoldable};
use middle::ty::MethodCall;

use syntax::ast::{self, Name};
use syntax::attr;
use syntax::codemap::DUMMY_SP;

use rustc_front::hir;

// drop_glue pointer, size, align.
const VTABLE_OFFSET: usize = 3;

/// The main "translation" pass for methods.  Generates code
/// for non-monomorphized methods only.  Other methods will
/// be generated once they are invoked with specific type parameters,
/// see `trans::base::lval_static_fn()` or `trans::base::monomorphic_fn()`.
pub fn trans_impl(ccx: &CrateContext,
                  name: ast::Name,
                  impl_items: &[hir::ImplItem],
                  generics: &hir::Generics,
                  id: ast::NodeId) {
    let _icx = push_ctxt("meth::trans_impl");
    let tcx = ccx.tcx();

    debug!("trans_impl(name={}, id={})", name, id);

    // Both here and below with generic methods, be sure to recurse and look for
    // items that we need to translate.
    if !generics.ty_params.is_empty() {
        return;
    }

    for impl_item in impl_items {
        match impl_item.node {
            hir::ImplItemKind::Method(ref sig, ref body) => {
                if sig.generics.ty_params.is_empty() {
                    let trans_everywhere = attr::requests_inline(&impl_item.attrs);
                    for (ref ccx, is_origin) in ccx.maybe_iter(trans_everywhere) {
                        let llfn = get_item_val(ccx, impl_item.id);
                        let empty_substs = tcx.mk_substs(Substs::trans_empty());
                        trans_fn(ccx,
                                 &sig.decl,
                                 body,
                                 llfn,
                                 empty_substs,
                                 impl_item.id,
                                 &impl_item.attrs);
                        update_linkage(ccx,
                                       llfn,
                                       Some(impl_item.id),
                                       if is_origin { OriginalTranslation } else { InlinedCopy });
                    }
                }
            }
            _ => {}
        }
    }
}

/// Compute the appropriate callee, give na method's ID, trait ID,
/// substitutions and a Vtable for that trait.
pub fn callee_for_trait_impl<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                       method_id: DefId,
                                       substs: &'tcx subst::Substs<'tcx>,
                                       trait_id: DefId,
                                       method_ty: Ty<'tcx>,
                                       vtable: traits::Vtable<'tcx, ()>)
                                       -> Callee<'tcx> {
    let _icx = push_ctxt("meth::callee_for_trait_impl");
    match vtable {
        traits::VtableImpl(vtable_impl) => {
            let impl_did = vtable_impl.impl_def_id;
            let mname = ccx.tcx().item_name(method_id);
            // create a concatenated set of substitutions which includes
            // those from the impl and those from the method:
            let impl_substs = vtable_impl.substs.with_method_from(&substs);
            let substs = ccx.tcx().mk_substs(impl_substs);
            let mth = get_impl_method(ccx.tcx(), impl_did, impl_substs, mname);

            // Translate the function, bypassing Callee::def.
            // That is because default methods have the same ID as the
            // trait method used to look up the impl method that ended
            // up here, so calling Callee::def would infinitely recurse.
            Callee::ptr(trans_fn_ref_with_substs(ccx, mth.method.def_id,
                                                 Some(method_ty), mth.substs))
        }
        traits::VtableClosure(vtable_closure) => {
            // The substitutions should have no type parameters remaining
            // after passing through fulfill_obligation
            let trait_closure_kind = ccx.tcx().lang_items.fn_trait_kind(trait_id).unwrap();
            let llfn = closure::trans_closure_method(ccx,
                                                     vtable_closure.closure_def_id,
                                                     vtable_closure.substs,
                                                     trait_closure_kind);
            let fn_ptr_ty = match method_ty.sty {
                ty::TyFnDef(_, _, fty) => ccx.tcx().mk_ty(ty::TyFnPtr(fty)),
                _ => unreachable!("expected fn item type, found {}",
                                  method_ty)
            };
            Callee::ptr(immediate_rvalue(llfn, fn_ptr_ty))
        }
        traits::VtableFnPointer(fn_ty) => {
            let trait_closure_kind = ccx.tcx().lang_items.fn_trait_kind(trait_id).unwrap();
            let llfn = trans_fn_pointer_shim(ccx, trait_closure_kind, fn_ty);
            let fn_ptr_ty = match method_ty.sty {
                ty::TyFnDef(_, _, fty) => ccx.tcx().mk_ty(ty::TyFnPtr(fty)),
                _ => unreachable!("expected fn item type, found {}",
                                  method_ty)
            };
            Callee::ptr(immediate_rvalue(llfn, fn_ptr_ty))
        }
        traits::VtableObject(ref data) => {
            Callee {
                data: Virtual(traits::get_vtable_index_of_object_method(
                    ccx.tcx(), data, method_id)),
                ty: method_ty
            }
        }
        traits::VtableBuiltin(..) |
        traits::VtableDefaultImpl(..) |
        traits::VtableParam(..) => {
            ccx.sess().bug(
                &format!("resolved vtable bad vtable {:?} in trans",
                        vtable));
        }
    }
}

/// Extracts a method from a trait object's vtable, at the
/// specified index, and casts it to the given type.
pub fn get_virtual_method<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                      llvtable: ValueRef,
                                      vtable_index: usize,
                                      method_ty: Ty<'tcx>)
                                      -> Datum<'tcx, Rvalue> {
    let _icx = push_ctxt("meth::get_virtual_method");
    let ccx = bcx.ccx();

    // Load the data pointer from the object.
    debug!("get_virtual_method(callee_ty={}, vtable_index={}, llvtable={})",
           method_ty,
           vtable_index,
           bcx.val_to_string(llvtable));

    let mptr = Load(bcx, GEPi(bcx, llvtable, &[vtable_index + VTABLE_OFFSET]));

    // Replace the self type (&Self or Box<Self>) with an opaque pointer.
    if let ty::TyFnDef(_, _, fty) = method_ty.sty {
        let opaque_ty = opaque_method_ty(ccx.tcx(), fty);
        immediate_rvalue(PointerCast(bcx, mptr, type_of(ccx, opaque_ty)), opaque_ty)
    } else {
        immediate_rvalue(mptr, method_ty)
    }
}

/// Generate a shim function that allows an object type like `SomeTrait` to
/// implement the type `SomeTrait`. Imagine a trait definition:
///
///    trait SomeTrait { fn get(&self) -> i32; ... }
///
/// And a generic bit of code:
///
///    fn foo<T:SomeTrait>(t: &T) {
///        let x = SomeTrait::get;
///        x(t)
///    }
///
/// What is the value of `x` when `foo` is invoked with `T=SomeTrait`?
/// The answer is that it is a shim function generated by this routine:
///
///    fn shim(t: &SomeTrait) -> i32 {
///        // ... call t.get() virtually ...
///    }
///
/// In fact, all virtual calls can be thought of as normal trait calls
/// that go through this shim function.
pub fn trans_object_shim<'a, 'tcx>(ccx: &'a CrateContext<'a, 'tcx>,
                                   method_ty: Ty<'tcx>,
                                   vtable_index: usize)
                                   -> Datum<'tcx, Rvalue> {
    let _icx = push_ctxt("trans_object_shim");
    let tcx = ccx.tcx();

    debug!("trans_object_shim(vtable_index={}, method_ty={:?})",
           vtable_index,
           method_ty);

    let ret_ty = tcx.erase_late_bound_regions(&method_ty.fn_ret());
    let ret_ty = infer::normalize_associated_type(tcx, &ret_ty);

    let shim_fn_ty = match method_ty.sty {
        ty::TyFnDef(_, _, fty) => tcx.mk_ty(ty::TyFnPtr(fty)),
        _ => unreachable!("expected fn item type, found {}", method_ty)
    };

    //
    let function_name = link::mangle_internal_name_by_type_and_seq(ccx, shim_fn_ty, "object_shim");
    let llfn = declare::define_internal_rust_fn(ccx, &function_name, shim_fn_ty);

    let empty_substs = tcx.mk_substs(Substs::trans_empty());
    let (block_arena, fcx): (TypedArena<_>, FunctionContext);
    block_arena = TypedArena::new();
    fcx = new_fn_ctxt(ccx,
                      llfn,
                      ast::DUMMY_NODE_ID,
                      false,
                      ret_ty,
                      empty_substs,
                      None,
                      &block_arena);
    let mut bcx = init_function(&fcx, false, ret_ty);

    let llargs = get_params(fcx.llfn);

    let self_idx = fcx.arg_offset();
    let llself = llargs[self_idx];
    let llvtable = llargs[self_idx + 1];

    debug!("trans_object_shim: llself={}, llvtable={}",
           bcx.val_to_string(llself), bcx.val_to_string(llvtable));

    assert!(!fcx.needs_ret_allocas);

    let dest =
        fcx.llretslotptr.get().map(
            |_| expr::SaveIn(fcx.get_ret_slot(bcx, ret_ty, "ret_slot")));

    debug!("trans_object_shim: method_offset_in_vtable={}",
           vtable_index);

    let callee = Callee {
        data: Virtual(vtable_index),
        ty: method_ty
    };
    bcx = callee.call(bcx, DebugLoc::None, ArgVals(&llargs[self_idx..]), dest).bcx;

    finish_fn(&fcx, bcx, ret_ty, DebugLoc::None);

    immediate_rvalue(llfn, shim_fn_ty)
}

/// Creates a returns a dynamic vtable for the given type and vtable origin.
/// This is used only for objects.
///
/// The `trait_ref` encodes the erased self type. Hence if we are
/// making an object `Foo<Trait>` from a value of type `Foo<T>`, then
/// `trait_ref` would map `T:Trait`.
pub fn get_vtable<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                            trait_ref: ty::PolyTraitRef<'tcx>)
                            -> ValueRef
{
    let tcx = ccx.tcx();
    let _icx = push_ctxt("meth::get_vtable");

    debug!("get_vtable(trait_ref={:?})", trait_ref);

    // Check the cache.
    match ccx.vtables().borrow().get(&trait_ref) {
        Some(&val) => { return val }
        None => { }
    }

    // Not in the cache. Build it.
    let methods = traits::supertraits(tcx, trait_ref.clone()).flat_map(|trait_ref| {
        let vtable = fulfill_obligation(ccx, DUMMY_SP, trait_ref.clone());
        match vtable {
            // Should default trait error here?
            traits::VtableDefaultImpl(_) |
            traits::VtableBuiltin(_) => {
                Vec::new().into_iter()
            }
            traits::VtableImpl(
                traits::VtableImplData {
                    impl_def_id: id,
                    substs,
                    nested: _ }) => {
                let nullptr = C_null(Type::nil(ccx).ptr_to());
                get_vtable_methods(ccx, id, substs)
                    .into_iter()
                    .map(|opt_mth| {
                        match opt_mth {
                            Some(mth) => {
                                trans_fn_ref_with_substs(ccx,
                                                         mth.method.def_id,
                                                         None,
                                                         mth.substs).val
                            }
                            None => nullptr
                        }
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
            }
            traits::VtableClosure(
                traits::VtableClosureData {
                    closure_def_id,
                    substs,
                    nested: _ }) => {
                let trait_closure_kind = tcx.lang_items.fn_trait_kind(trait_ref.def_id()).unwrap();
                let llfn = closure::trans_closure_method(ccx,
                                                         closure_def_id,
                                                         substs,
                                                         trait_closure_kind);
                vec![llfn].into_iter()
            }
            traits::VtableFnPointer(bare_fn_ty) => {
                let trait_closure_kind = tcx.lang_items.fn_trait_kind(trait_ref.def_id()).unwrap();
                vec![trans_fn_pointer_shim(ccx, trait_closure_kind, bare_fn_ty)].into_iter()
            }
            traits::VtableObject(ref data) => {
                // this would imply that the Self type being erased is
                // an object type; this cannot happen because we
                // cannot cast an unsized type into a trait object
                tcx.sess.bug(
                    &format!("cannot get vtable for an object type: {:?}",
                            data));
            }
            traits::VtableParam(..) => {
                tcx.sess.bug(
                    &format!("resolved vtable for {:?} to bad vtable {:?} in trans",
                            trait_ref,
                            vtable));
            }
        }
    });

    let size_ty = sizing_type_of(ccx, trait_ref.self_ty());
    let size = machine::llsize_of_alloc(ccx, size_ty);
    let align = align_of(ccx, trait_ref.self_ty());

    let components: Vec<_> = vec![
        // Generate a destructor for the vtable.
        glue::get_drop_glue(ccx, trait_ref.self_ty()),
        C_uint(ccx, size),
        C_uint(ccx, align)
    ].into_iter().chain(methods).collect();

    let vtable_const = C_struct(ccx, &components, false);
    let align = machine::llalign_of_pref(ccx, val_ty(vtable_const));
    let vtable = consts::addr_of(ccx, vtable_const, align, "vtable");

    ccx.vtables().borrow_mut().insert(trait_ref, vtable);
    vtable
}

pub fn get_vtable_methods<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                    impl_id: DefId,
                                    substs: &'tcx subst::Substs<'tcx>)
                                    -> Vec<Option<ImplMethod<'tcx>>>
{
    let tcx = ccx.tcx();

    debug!("get_vtable_methods(impl_id={:?}, substs={:?}", impl_id, substs);

    let trt_id = match tcx.impl_trait_ref(impl_id) {
        Some(t_id) => t_id.def_id,
        None       => ccx.sess().bug("make_impl_vtable: don't know how to \
                                      make a vtable for a type impl!")
    };

    tcx.populate_implementations_for_trait_if_necessary(trt_id);

    let trait_item_def_ids = tcx.trait_item_def_ids(trt_id);
    trait_item_def_ids
        .iter()

        // Filter out non-method items.
        .filter_map(|item_def_id| {
            match *item_def_id {
                ty::MethodTraitItemId(def_id) => Some(def_id),
                _ => None,
            }
        })

        // Now produce pointers for each remaining method. If the
        // method could never be called from this object, just supply
        // null.
        .map(|trait_method_def_id| {
            debug!("get_vtable_methods: trait_method_def_id={:?}",
                   trait_method_def_id);

            let trait_method_type = match tcx.impl_or_trait_item(trait_method_def_id) {
                ty::MethodTraitItem(m) => m,
                _ => ccx.sess().bug("should be a method, not other assoc item"),
            };
            let name = trait_method_type.name;

            // Some methods cannot be called on an object; skip those.
            if !traits::is_vtable_safe_method(tcx, trt_id, &trait_method_type) {
                debug!("get_vtable_methods: not vtable safe");
                return None;
            }

            debug!("get_vtable_methods: trait_method_type={:?}",
                   trait_method_type);

            // The substitutions we have are on the impl, so we grab
            // the method type from the impl to substitute into.
            let mth = get_impl_method(tcx, impl_id, substs.clone(), name);

            debug!("get_vtable_methods: mth={:?}", mth);

            // If this is a default method, it's possible that it
            // relies on where clauses that do not hold for this
            // particular set of type parameters. Note that this
            // method could then never be called, so we do not want to
            // try and trans it, in that case. Issue #23435.
            if mth.is_provided {
                let predicates = mth.method.predicates.predicates.subst(tcx, mth.substs);
                if !normalize_and_test_predicates(ccx, predicates.into_vec()) {
                    debug!("get_vtable_methods: predicates do not hold");
                    return None;
                }
            }

            Some(mth)
        })
        .collect()
}

/// Replace the self type (&Self or Box<Self>) with an opaque pointer.
fn opaque_method_ty<'tcx>(tcx: &TyCtxt<'tcx>, method_ty: &ty::BareFnTy<'tcx>)
                          -> Ty<'tcx> {
    let mut inputs = method_ty.sig.0.inputs.clone();
    inputs[0] = tcx.mk_mut_ptr(tcx.mk_mach_int(ast::IntTy::I8));

    tcx.mk_fn_ptr(ty::BareFnTy {
        unsafety: method_ty.unsafety,
        abi: method_ty.abi,
        sig: ty::Binder(ty::FnSig {
            inputs: inputs,
            output: method_ty.sig.0.output,
            variadic: method_ty.sig.0.variadic,
        }),
    })
}

#[derive(Debug)]
pub struct ImplMethod<'tcx> {
    pub method: Rc<ty::Method<'tcx>>,
    pub substs: Substs<'tcx>,
    pub is_provided: bool
}

/// Locates the applicable definition of a method, given its name.
pub fn get_impl_method<'tcx>(tcx: &ty::ctxt<'tcx>,
                             impl_def_id: DefId,
                             substs: Substs<'tcx>,
                             name: Name)
                             -> ImplMethod<'tcx>
{
    assert!(!substs.types.needs_infer());

    let trait_def_id = tcx.trait_id_of_impl(impl_def_id).unwrap();
    let trait_def = tcx.lookup_trait_def(trait_def_id);
    let infcx = infer::normalizing_infer_ctxt(tcx, &tcx.tables);

    match trait_def.ancestors(impl_def_id).fn_defs(tcx, name).next() {
        Some(node_item) => {
            ImplMethod {
                method: node_item.item,
                substs: traits::translate_substs(&infcx, impl_def_id, substs, node_item.node),
                is_provided: node_item.node.is_from_trait(),
            }
        }
        None => {
            tcx.sess.bug(&format!("method {:?} not found in {:?}", name, impl_def_id))
        }
    }
}
