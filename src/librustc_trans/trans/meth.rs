// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use arena::TypedArena;
use back::link;
use llvm::{ValueRef, get_params};
use middle::def_id::DefId;
use middle::infer;
use middle::subst::{Subst, Substs};
use middle::subst::VecPerParamSpace;
use middle::subst;
use middle::traits;
use trans::base::*;
use trans::build::*;
use trans::callee::*;
use trans::callee;
use trans::cleanup;
use trans::closure;
use trans::common::*;
use trans::consts;
use trans::datum::*;
use trans::debuginfo::DebugLoc;
use trans::declare;
use trans::expr::SaveIn;
use trans::expr;
use trans::glue;
use trans::machine;
use trans::monomorphize;
use trans::type_::Type;
use trans::type_of::*;
use middle::ty::{self, Ty, HasTypeFlags};
use middle::ty::MethodCall;

use syntax::ast;
use syntax::attr;
use syntax::codemap::DUMMY_SP;
use syntax::ptr::P;

use rustc_front::hir;

// drop_glue pointer, size, align.
const VTABLE_OFFSET: usize = 3;

/// The main "translation" pass for methods.  Generates code
/// for non-monomorphized methods only.  Other methods will
/// be generated once they are invoked with specific type parameters,
/// see `trans::base::lval_static_fn()` or `trans::base::monomorphic_fn()`.
pub fn trans_impl(ccx: &CrateContext,
                  name: ast::Name,
                  impl_items: &[P<hir::ImplItem>],
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
                        trans_fn(ccx, &sig.decl, body, llfn,
                                 empty_substs, impl_item.id, &[]);
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

pub fn trans_method_callee<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                       method_call: MethodCall,
                                       self_expr: Option<&hir::Expr>,
                                       arg_cleanup_scope: cleanup::ScopeId)
                                       -> Callee<'blk, 'tcx> {
    let _icx = push_ctxt("meth::trans_method_callee");

    let method = bcx.tcx().tables.borrow().method_map[&method_call];

    match bcx.tcx().impl_or_trait_item(method.def_id).container() {
        ty::ImplContainer(_) => {
            debug!("trans_method_callee: static, {:?}", method.def_id);
            let datum = callee::trans_fn_ref(bcx.ccx(),
                                             method.def_id,
                                             MethodCallKey(method_call),
                                             bcx.fcx.param_substs);
            Callee {
                bcx: bcx,
                data: Fn(datum.val),
                ty: datum.ty
            }
        }

        ty::TraitContainer(trait_def_id) => {
            let trait_substs = method.substs.clone().method_to_trait();
            let trait_substs = bcx.tcx().mk_substs(trait_substs);
            let trait_ref = ty::TraitRef::new(trait_def_id, trait_substs);

            let trait_ref = ty::Binder(bcx.monomorphize(&trait_ref));
            let span = bcx.tcx().map.span(method_call.expr_id);
            debug!("method_call={:?} trait_ref={:?} trait_ref id={:?} substs={:?}",
                   method_call,
                   trait_ref,
                   trait_ref.0.def_id,
                   trait_ref.0.substs);
            let origin = fulfill_obligation(bcx.ccx(),
                                            span,
                                            trait_ref.clone());
            debug!("origin = {:?}", origin);
            trans_monomorphized_callee(bcx,
                                       method_call,
                                       self_expr,
                                       trait_def_id,
                                       method.def_id,
                                       method.ty,
                                       origin,
                                       arg_cleanup_scope)
        }
    }
}

pub fn trans_static_method_callee<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                            method_id: DefId,
                                            trait_id: DefId,
                                            expr_id: ast::NodeId,
                                            param_substs: &'tcx subst::Substs<'tcx>)
                                            -> Datum<'tcx, Rvalue>
{
    let _icx = push_ctxt("meth::trans_static_method_callee");
    let tcx = ccx.tcx();

    debug!("trans_static_method_callee(method_id={:?}, trait_id={}, \
            expr_id={})",
           method_id,
           tcx.item_path_str(trait_id),
           expr_id);

    let mname = tcx.item_name(method_id);

    debug!("trans_static_method_callee: method_id={:?}, expr_id={}, \
            name={}", method_id, expr_id, mname);

    // Find the substitutions for the fn itself. This includes
    // type parameters that belong to the trait but also some that
    // belong to the method:
    let rcvr_substs = node_id_substs(ccx, ExprId(expr_id), param_substs);
    let subst::SeparateVecsPerParamSpace {
        types: rcvr_type,
        selfs: rcvr_self,
        fns: rcvr_method
    } = rcvr_substs.types.split();

    // Lookup the precise impl being called. To do that, we need to
    // create a trait reference identifying the self type and other
    // input type parameters. To create that trait reference, we have
    // to pick apart the type parameters to identify just those that
    // pertain to the trait. This is easiest to explain by example:
    //
    //     trait Convert {
    //         fn from<U:Foo>(n: U) -> Option<Self>;
    //     }
    //     ...
    //     let f = <Vec<i32> as Convert>::from::<String>(...)
    //
    // Here, in this call, which I've written with explicit UFCS
    // notation, the set of type parameters will be:
    //
    //     rcvr_type: [] <-- nothing declared on the trait itself
    //     rcvr_self: [Vec<i32>] <-- the self type
    //     rcvr_method: [String] <-- method type parameter
    //
    // So we create a trait reference using the first two,
    // basically corresponding to `<Vec<i32> as Convert>`.
    // The remaining type parameters (`rcvr_method`) will be used below.
    let trait_substs =
        Substs::erased(VecPerParamSpace::new(rcvr_type,
                                             rcvr_self,
                                             Vec::new()));
    let trait_substs = tcx.mk_substs(trait_substs);
    debug!("trait_substs={:?}", trait_substs);
    let trait_ref = ty::Binder(ty::TraitRef::new(trait_id, trait_substs));
    let vtbl = fulfill_obligation(ccx,
                                  DUMMY_SP,
                                  trait_ref);

    // Now that we know which impl is being used, we can dispatch to
    // the actual function:
    match vtbl {
        traits::VtableImpl(traits::VtableImplData {
            impl_def_id: impl_did,
            substs: impl_substs,
            nested: _ }) =>
        {
            assert!(!impl_substs.types.needs_infer());

            // Create the substitutions that are in scope. This combines
            // the type parameters from the impl with those declared earlier.
            // To see what I mean, consider a possible impl:
            //
            //    impl<T> Convert for Vec<T> {
            //        fn from<U:Foo>(n: U) { ... }
            //    }
            //
            // Recall that we matched `<Vec<i32> as Convert>`. Trait
            // resolution will have given us a substitution
            // containing `impl_substs=[[T=i32],[],[]]` (the type
            // parameters defined on the impl). We combine
            // that with the `rcvr_method` from before, which tells us
            // the type parameters from the *method*, to yield
            // `callee_substs=[[T=i32],[],[U=String]]`.
            let subst::SeparateVecsPerParamSpace {
                types: impl_type,
                selfs: impl_self,
                fns: _
            } = impl_substs.types.split();
            let callee_substs =
                Substs::erased(VecPerParamSpace::new(impl_type,
                                                     impl_self,
                                                     rcvr_method));

            let mth = tcx.get_impl_method(impl_did, callee_substs, mname);
            trans_fn_ref_with_substs(ccx, mth.method.def_id, ExprId(expr_id),
                                     param_substs,
                                     mth.substs)
        }
        traits::VtableObject(ref data) => {
            let idx = traits::get_vtable_index_of_object_method(tcx, data, method_id);
            trans_object_shim(ccx,
                              data.upcast_trait_ref.clone(),
                              method_id,
                              idx)
        }
        _ => {
            tcx.sess.bug(&format!("static call to invalid vtable: {:?}",
                                 vtbl));
        }
    }
}

fn trans_monomorphized_callee<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                          method_call: MethodCall,
                                          self_expr: Option<&hir::Expr>,
                                          trait_id: DefId,
                                          method_id: DefId,
                                          method_ty: Ty<'tcx>,
                                          vtable: traits::Vtable<'tcx, ()>,
                                          arg_cleanup_scope: cleanup::ScopeId)
                                          -> Callee<'blk, 'tcx> {
    let _icx = push_ctxt("meth::trans_monomorphized_callee");
    match vtable {
        traits::VtableImpl(vtable_impl) => {
            let ccx = bcx.ccx();
            let impl_did = vtable_impl.impl_def_id;
            let mname = match ccx.tcx().impl_or_trait_item(method_id) {
                ty::MethodTraitItem(method) => method.name,
                _ => {
                    bcx.tcx().sess.bug("can't monomorphize a non-method trait \
                                        item")
                }
            };
            // create a concatenated set of substitutions which includes
            // those from the impl and those from the method:
            let callee_substs =
                combine_impl_and_methods_tps(
                    bcx, MethodCallKey(method_call), vtable_impl.substs);

            let mth = bcx.tcx().get_impl_method(impl_did, callee_substs, mname);
            // translate the function
            let datum = trans_fn_ref_with_substs(bcx.ccx(),
                                                 mth.method.def_id,
                                                 MethodCallKey(method_call),
                                                 bcx.fcx.param_substs,
                                                 mth.substs);

            Callee { bcx: bcx, data: Fn(datum.val), ty: datum.ty }
        }
        traits::VtableClosure(vtable_closure) => {
            // The substitutions should have no type parameters remaining
            // after passing through fulfill_obligation
            let trait_closure_kind = bcx.tcx().lang_items.fn_trait_kind(trait_id).unwrap();
            let llfn = closure::trans_closure_method(bcx.ccx(),
                                                     vtable_closure.closure_def_id,
                                                     vtable_closure.substs,
                                                     trait_closure_kind);
            Callee {
                bcx: bcx,
                data: Fn(llfn),
                ty: monomorphize_type(bcx, method_ty)
            }
        }
        traits::VtableFnPointer(fn_ty) => {
            let trait_closure_kind = bcx.tcx().lang_items.fn_trait_kind(trait_id).unwrap();
            let llfn = trans_fn_pointer_shim(bcx.ccx(), trait_closure_kind, fn_ty);
            Callee {
                bcx: bcx,
                data: Fn(llfn),
                ty: monomorphize_type(bcx, method_ty)
            }
        }
        traits::VtableObject(ref data) => {
            let idx = traits::get_vtable_index_of_object_method(bcx.tcx(), data, method_id);
            if let Some(self_expr) = self_expr {
                if let ty::TyBareFn(_, ref fty) = monomorphize_type(bcx, method_ty).sty {
                    let ty = bcx.tcx().mk_fn(None, opaque_method_ty(bcx.tcx(), fty));
                    return trans_trait_callee(bcx, ty, idx, self_expr, arg_cleanup_scope);
                }
            }
            let datum = trans_object_shim(bcx.ccx(),
                                          data.upcast_trait_ref.clone(),
                                          method_id,
                                          idx);
            Callee { bcx: bcx, data: Fn(datum.val), ty: datum.ty }
        }
        traits::VtableBuiltin(..) |
        traits::VtableDefaultImpl(..) |
        traits::VtableParam(..) => {
            bcx.sess().bug(
                &format!("resolved vtable bad vtable {:?} in trans",
                        vtable));
        }
    }
}

 /// Creates a concatenated set of substitutions which includes those from the impl and those from
 /// the method.  This are some subtle complications here.  Statically, we have a list of type
 /// parameters like `[T0, T1, T2, M1, M2, M3]` where `Tn` are type parameters that appear on the
 /// receiver.  For example, if the receiver is a method parameter `A` with a bound like
 /// `trait<B,C,D>` then `Tn` would be `[B,C,D]`.
 ///
 /// The weird part is that the type `A` might now be bound to any other type, such as `foo<X>`.
 /// In that case, the vector we want is: `[X, M1, M2, M3]`.  Therefore, what we do now is to slice
 /// off the method type parameters and append them to the type parameters from the type that the
 /// receiver is mapped to.
fn combine_impl_and_methods_tps<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                            node: ExprOrMethodCall,
                                            rcvr_substs: subst::Substs<'tcx>)
                                            -> subst::Substs<'tcx>
{
    let ccx = bcx.ccx();

    let node_substs = node_id_substs(ccx, node, bcx.fcx.param_substs);

    debug!("rcvr_substs={:?}", rcvr_substs);
    debug!("node_substs={:?}", node_substs);

    // Break apart the type parameters from the node and type
    // parameters from the receiver.
    let node_method = node_substs.types.split().fns;
    let subst::SeparateVecsPerParamSpace {
        types: rcvr_type,
        selfs: rcvr_self,
        fns: rcvr_method
    } = rcvr_substs.types.clone().split();
    assert!(rcvr_method.is_empty());
    subst::Substs {
        regions: subst::ErasedRegions,
        types: subst::VecPerParamSpace::new(rcvr_type, rcvr_self, node_method)
    }
}

/// Create a method callee where the method is coming from a trait object (e.g., Box<Trait> type).
/// In this case, we must pull the fn pointer out of the vtable that is packaged up with the
/// object. Objects are represented as a pair, so we first evaluate the self expression and then
/// extract the self data and vtable out of the pair.
fn trans_trait_callee<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                  opaque_fn_ty: Ty<'tcx>,
                                  vtable_index: usize,
                                  self_expr: &hir::Expr,
                                  arg_cleanup_scope: cleanup::ScopeId)
                                  -> Callee<'blk, 'tcx> {
    let _icx = push_ctxt("meth::trans_trait_callee");
    let mut bcx = bcx;

    // Translate self_datum and take ownership of the value by
    // converting to an rvalue.
    let self_datum = unpack_datum!(
        bcx, expr::trans(bcx, self_expr));

    let llval = if bcx.fcx.type_needs_drop(self_datum.ty) {
        let self_datum = unpack_datum!(
            bcx, self_datum.to_rvalue_datum(bcx, "trait_callee"));

        // Convert to by-ref since `trans_trait_callee_from_llval` wants it
        // that way.
        let self_datum = unpack_datum!(
            bcx, self_datum.to_ref_datum(bcx));

        // Arrange cleanup in case something should go wrong before the
        // actual call occurs.
        self_datum.add_clean(bcx.fcx, arg_cleanup_scope)
    } else {
        // We don't have to do anything about cleanups for &Trait and &mut Trait.
        assert!(self_datum.kind.is_by_ref());
        self_datum.val
    };

    let llself = Load(bcx, expr::get_dataptr(bcx, llval));
    let llvtable = Load(bcx, expr::get_meta(bcx, llval));
    trans_trait_callee_from_llval(bcx, opaque_fn_ty, vtable_index, llself, llvtable)
}

/// Same as `trans_trait_callee()` above, except that it is given a by-ref pointer to the object
/// pair.
fn trans_trait_callee_from_llval<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                             opaque_fn_ty: Ty<'tcx>,
                                             vtable_index: usize,
                                             llself: ValueRef,
                                             llvtable: ValueRef)
                                             -> Callee<'blk, 'tcx> {
    let _icx = push_ctxt("meth::trans_trait_callee");
    let ccx = bcx.ccx();

    // Load the data pointer from the object.
    debug!("trans_trait_callee_from_llval(callee_ty={}, vtable_index={}, llself={}, llvtable={})",
           opaque_fn_ty,
           vtable_index,
           bcx.val_to_string(llself),
           bcx.val_to_string(llvtable));

    // Replace the self type (&Self or Box<Self>) with an opaque pointer.
    let mptr = Load(bcx, GEPi(bcx, llvtable, &[vtable_index + VTABLE_OFFSET]));
    let llcallee_ty = type_of_fn_from_ty(ccx, opaque_fn_ty);

    Callee {
        bcx: bcx,
        data: TraitItem(MethodData {
            llfn: PointerCast(bcx, mptr, llcallee_ty.ptr_to()),
            llself: PointerCast(bcx, llself, Type::i8p(ccx)),
        }),
        ty: opaque_fn_ty
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
fn trans_object_shim<'a, 'tcx>(
    ccx: &'a CrateContext<'a, 'tcx>,
    upcast_trait_ref: ty::PolyTraitRef<'tcx>,
    method_id: DefId,
    vtable_index: usize)
    -> Datum<'tcx, Rvalue>
{
    let _icx = push_ctxt("trans_object_shim");
    let tcx = ccx.tcx();

    debug!("trans_object_shim(upcast_trait_ref={:?}, method_id={:?})",
           upcast_trait_ref,
           method_id);

    // Upcast to the trait in question and extract out the substitutions.
    let upcast_trait_ref = tcx.erase_late_bound_regions(&upcast_trait_ref);
    let object_substs = upcast_trait_ref.substs.clone().erase_regions();
    debug!("trans_object_shim: object_substs={:?}", object_substs);

    // Lookup the type of this method as declared in the trait and apply substitutions.
    let method_ty = match tcx.impl_or_trait_item(method_id) {
        ty::MethodTraitItem(method) => method,
        _ => {
            tcx.sess.bug("can't create a method shim for a non-method item")
        }
    };
    let fty = monomorphize::apply_param_substs(tcx, &object_substs, &method_ty.fty);
    let fty = tcx.mk_bare_fn(fty);
    let method_ty = opaque_method_ty(tcx, fty);
    debug!("trans_object_shim: fty={:?} method_ty={:?}", fty, method_ty);

    //
    let shim_fn_ty = tcx.mk_fn(None, fty);
    let method_bare_fn_ty = tcx.mk_fn(None, method_ty);
    let function_name = link::mangle_internal_name_by_type_and_seq(ccx, shim_fn_ty, "object_shim");
    let llfn = declare::define_internal_rust_fn(ccx, &function_name, shim_fn_ty);

    let sig = ccx.tcx().erase_late_bound_regions(&fty.sig);
    let sig = infer::normalize_associated_type(ccx.tcx(), &sig);

    let empty_substs = tcx.mk_substs(Substs::trans_empty());
    let (block_arena, fcx): (TypedArena<_>, FunctionContext);
    block_arena = TypedArena::new();
    fcx = new_fn_ctxt(ccx,
                      llfn,
                      ast::DUMMY_NODE_ID,
                      false,
                      sig.output,
                      empty_substs,
                      None,
                      &block_arena);
    let mut bcx = init_function(&fcx, false, sig.output);

    let llargs = get_params(fcx.llfn);

    let self_idx = fcx.arg_offset();
    let llself = llargs[self_idx];
    let llvtable = llargs[self_idx + 1];

    debug!("trans_object_shim: llself={}, llvtable={}",
           bcx.val_to_string(llself), bcx.val_to_string(llvtable));

    assert!(!fcx.needs_ret_allocas);

    let dest =
        fcx.llretslotptr.get().map(
            |_| expr::SaveIn(fcx.get_ret_slot(bcx, sig.output, "ret_slot")));

    debug!("trans_object_shim: method_offset_in_vtable={}",
           vtable_index);

    bcx = trans_call_inner(bcx,
                           DebugLoc::None,
                           |bcx, _| trans_trait_callee_from_llval(bcx,
                                                                  method_bare_fn_ty,
                                                                  vtable_index,
                                                                  llself, llvtable),
                           ArgVals(&llargs[(self_idx + 2)..]),
                           dest).bcx;

    finish_fn(&fcx, bcx, sig.output, DebugLoc::None);

    immediate_rvalue(llfn, shim_fn_ty)
}

/// Creates a returns a dynamic vtable for the given type and vtable origin.
/// This is used only for objects.
///
/// The `trait_ref` encodes the erased self type. Hence if we are
/// making an object `Foo<Trait>` from a value of type `Foo<T>`, then
/// `trait_ref` would map `T:Trait`.
pub fn get_vtable<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                            trait_ref: ty::PolyTraitRef<'tcx>,
                            param_substs: &'tcx subst::Substs<'tcx>)
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
                emit_vtable_methods(ccx, id, substs, param_substs).into_iter()
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

fn emit_vtable_methods<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                 impl_id: DefId,
                                 substs: subst::Substs<'tcx>,
                                 param_substs: &'tcx subst::Substs<'tcx>)
                                 -> Vec<ValueRef>
{
    let tcx = ccx.tcx();

    debug!("emit_vtable_methods(impl_id={:?}, substs={:?}, param_substs={:?})",
           impl_id,
           substs,
           param_substs);

    let trt_id = match tcx.impl_trait_ref(impl_id) {
        Some(t_id) => t_id.def_id,
        None       => ccx.sess().bug("make_impl_vtable: don't know how to \
                                      make a vtable for a type impl!")
    };

    tcx.populate_implementations_for_trait_if_necessary(trt_id);

    let nullptr = C_null(Type::nil(ccx).ptr_to());
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
            debug!("emit_vtable_methods: trait_method_def_id={:?}",
                   trait_method_def_id);

            let trait_method_type = match tcx.impl_or_trait_item(trait_method_def_id) {
                ty::MethodTraitItem(m) => m,
                _ => ccx.sess().bug("should be a method, not other assoc item"),
            };
            let name = trait_method_type.name;

            // Some methods cannot be called on an object; skip those.
            if !traits::is_vtable_safe_method(tcx, trt_id, &trait_method_type) {
                debug!("emit_vtable_methods: not vtable safe");
                return nullptr;
            }

            debug!("emit_vtable_methods: trait_method_type={:?}",
                   trait_method_type);

            // The substitutions we have are on the impl, so we grab
            // the method type from the impl to substitute into.
            let mth = tcx.get_impl_method(impl_id, substs.clone(), name);

            debug!("emit_vtable_methods: mth={:?}", mth);

            // If this is a default method, it's possible that it
            // relies on where clauses that do not hold for this
            // particular set of type parameters. Note that this
            // method could then never be called, so we do not want to
            // try and trans it, in that case. Issue #23435.
            if mth.is_provided {
                let predicates = mth.method.predicates.predicates.subst(tcx, &mth.substs);
                if !normalize_and_test_predicates(ccx, predicates.into_vec()) {
                    debug!("emit_vtable_methods: predicates do not hold");
                    return nullptr;
                }
            }

            trans_fn_ref_with_substs(ccx,
                                     mth.method.def_id,
                                     ExprId(0),
                                     param_substs,
                                     mth.substs).val
        })
        .collect()
}

/// Replace the self type (&Self or Box<Self>) with an opaque pointer.
fn opaque_method_ty<'tcx>(tcx: &ty::ctxt<'tcx>, method_ty: &ty::BareFnTy<'tcx>)
                          -> &'tcx ty::BareFnTy<'tcx> {
    let mut inputs = method_ty.sig.0.inputs.clone();
    inputs[0] = tcx.mk_mut_ptr(tcx.mk_mach_int(ast::TyI8));

    tcx.mk_bare_fn(ty::BareFnTy {
        unsafety: method_ty.unsafety,
        abi: method_ty.abi,
        sig: ty::Binder(ty::FnSig {
            inputs: inputs,
            output: method_ty.sig.0.output,
            variadic: method_ty.sig.0.variadic,
        }),
    })
}
