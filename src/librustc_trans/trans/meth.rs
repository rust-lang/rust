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
use back::abi;
use back::link;
use llvm::{ValueRef, get_param};
use metadata::csearch;
use middle::subst::Substs;
use middle::subst::VecPerParamSpace;
use middle::subst;
use middle::traits;
use trans::base::*;
use trans::build::*;
use trans::callee::*;
use trans::callee;
use trans::cleanup;
use trans::common::*;
use trans::consts;
use trans::datum::*;
use trans::debuginfo::DebugLoc;
use trans::expr::SaveIn;
use trans::expr;
use trans::glue;
use trans::machine;
use trans::monomorphize;
use trans::type_::Type;
use trans::type_of::*;
use middle::ty::{self, Ty};
use middle::ty::MethodCall;
use util::ppaux::Repr;

use std::rc::Rc;
use syntax::abi::{Rust, RustCall};
use syntax::parse::token;
use syntax::{ast, ast_map, attr, visit};
use syntax::codemap::DUMMY_SP;
use syntax::ptr::P;

// drop_glue pointer, size, align.
const VTABLE_OFFSET: uint = 3;

/// The main "translation" pass for methods.  Generates code
/// for non-monomorphized methods only.  Other methods will
/// be generated once they are invoked with specific type parameters,
/// see `trans::base::lval_static_fn()` or `trans::base::monomorphic_fn()`.
pub fn trans_impl(ccx: &CrateContext,
                  name: ast::Ident,
                  impl_items: &[P<ast::ImplItem>],
                  generics: &ast::Generics,
                  id: ast::NodeId) {
    let _icx = push_ctxt("meth::trans_impl");
    let tcx = ccx.tcx();

    debug!("trans_impl(name={}, id={})", name.repr(tcx), id);

    let mut v = TransItemVisitor { ccx: ccx };

    // Both here and below with generic methods, be sure to recurse and look for
    // items that we need to translate.
    if !generics.ty_params.is_empty() {
        for impl_item in impl_items {
            match impl_item.node {
                ast::MethodImplItem(..) => {
                    visit::walk_impl_item(&mut v, impl_item);
                }
                ast::TypeImplItem(_) |
                ast::MacImplItem(_) => {}
            }
        }
        return;
    }
    for impl_item in impl_items {
        match impl_item.node {
            ast::MethodImplItem(ref sig, ref body) => {
                if sig.generics.ty_params.len() == 0 {
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
                visit::walk_impl_item(&mut v, impl_item);
            }
            ast::TypeImplItem(_) |
            ast::MacImplItem(_) => {}
        }
    }
}

pub fn trans_method_callee<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                       method_call: MethodCall,
                                       self_expr: Option<&ast::Expr>,
                                       arg_cleanup_scope: cleanup::ScopeId)
                                       -> Callee<'blk, 'tcx> {
    let _icx = push_ctxt("meth::trans_method_callee");

    let (origin, method_ty) =
        bcx.tcx().method_map
                 .borrow()
                 .get(&method_call)
                 .map(|method| (method.origin.clone(), method.ty))
                 .unwrap();

    match origin {
        ty::MethodStatic(did) |
        ty::MethodStaticClosure(did) => {
            Callee {
                bcx: bcx,
                data: Fn(callee::trans_fn_ref(bcx.ccx(),
                                              did,
                                              MethodCallKey(method_call),
                                              bcx.fcx.param_substs).val),
            }
        }

        ty::MethodTypeParam(ty::MethodParam {
            ref trait_ref,
            method_num,
            impl_def_id: _
        }) => {
            let trait_ref = ty::Binder(bcx.monomorphize(trait_ref));
            let span = bcx.tcx().map.span(method_call.expr_id);
            debug!("method_call={:?} trait_ref={}",
                   method_call,
                   trait_ref.repr(bcx.tcx()));
            let origin = fulfill_obligation(bcx.ccx(),
                                            span,
                                            trait_ref.clone());
            debug!("origin = {}", origin.repr(bcx.tcx()));
            trans_monomorphized_callee(bcx,
                                       method_call,
                                       trait_ref.def_id(),
                                       method_num,
                                       origin)
        }

        ty::MethodTraitObject(ref mt) => {
            let self_expr = match self_expr {
                Some(self_expr) => self_expr,
                None => {
                    bcx.sess().span_bug(bcx.tcx().map.span(method_call.expr_id),
                                        "self expr wasn't provided for trait object \
                                         callee (trying to call overloaded op?)")
                }
            };
            trans_trait_callee(bcx,
                               monomorphize_type(bcx, method_ty),
                               mt.vtable_index,
                               self_expr,
                               arg_cleanup_scope)
        }
    }
}

pub fn trans_static_method_callee<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                            method_id: ast::DefId,
                                            trait_id: ast::DefId,
                                            expr_id: ast::NodeId,
                                            param_substs: &'tcx subst::Substs<'tcx>)
                                            -> Datum<'tcx, Rvalue>
{
    let _icx = push_ctxt("meth::trans_static_method_callee");
    let tcx = ccx.tcx();

    debug!("trans_static_method_callee(method_id={:?}, trait_id={}, \
            expr_id={})",
           method_id,
           ty::item_path_str(tcx, trait_id),
           expr_id);

    let mname = if method_id.krate == ast::LOCAL_CRATE {
        match tcx.map.get(method_id.node) {
            ast_map::NodeTraitItem(trait_item) => trait_item.ident.name,
            _ => panic!("callee is not a trait method")
        }
    } else {
        csearch::get_item_path(tcx, method_id).last().unwrap().name()
    };
    debug!("trans_static_method_callee: method_id={:?}, expr_id={}, \
            name={}", method_id, expr_id, token::get_name(mname));

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
    //     let f = <Vec<int> as Convert>::from::<String>(...)
    //
    // Here, in this call, which I've written with explicit UFCS
    // notation, the set of type parameters will be:
    //
    //     rcvr_type: [] <-- nothing declared on the trait itself
    //     rcvr_self: [Vec<int>] <-- the self type
    //     rcvr_method: [String] <-- method type parameter
    //
    // So we create a trait reference using the first two,
    // basically corresponding to `<Vec<int> as Convert>`.
    // The remaining type parameters (`rcvr_method`) will be used below.
    let trait_substs =
        Substs::erased(VecPerParamSpace::new(rcvr_type,
                                             rcvr_self,
                                             Vec::new()));
    let trait_substs = tcx.mk_substs(trait_substs);
    debug!("trait_substs={}", trait_substs.repr(tcx));
    let trait_ref = ty::Binder(Rc::new(ty::TraitRef { def_id: trait_id,
                                                      substs: trait_substs }));
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
            assert!(impl_substs.types.all(|t| !ty::type_needs_infer(*t)));

            // Create the substitutions that are in scope. This combines
            // the type parameters from the impl with those declared earlier.
            // To see what I mean, consider a possible impl:
            //
            //    impl<T> Convert for Vec<T> {
            //        fn from<U:Foo>(n: U) { ... }
            //    }
            //
            // Recall that we matched `<Vec<int> as Convert>`. Trait
            // resolution will have given us a substitution
            // containing `impl_substs=[[T=int],[],[]]` (the type
            // parameters defined on the impl). We combine
            // that with the `rcvr_method` from before, which tells us
            // the type parameters from the *method*, to yield
            // `callee_substs=[[T=int],[],[U=String]]`.
            let subst::SeparateVecsPerParamSpace {
                types: impl_type,
                selfs: impl_self,
                fns: _
            } = impl_substs.types.split();
            let callee_substs =
                Substs::erased(VecPerParamSpace::new(impl_type,
                                                     impl_self,
                                                     rcvr_method));

            let mth_id = method_with_name(ccx, impl_did, mname);
            trans_fn_ref_with_substs(ccx, mth_id, ExprId(expr_id),
                                     param_substs,
                                     callee_substs)
        }
        traits::VtableObject(ref data) => {
            let trait_item_def_ids =
                ty::trait_item_def_ids(ccx.tcx(), trait_id);
            let method_offset_in_trait =
                trait_item_def_ids.iter()
                                  .position(|item| item.def_id() == method_id)
                                  .unwrap();
            let (llfn, ty) =
                trans_object_shim(ccx,
                                  data.object_ty,
                                  data.upcast_trait_ref.clone(),
                                  method_offset_in_trait);
            immediate_rvalue(llfn, ty)
        }
        _ => {
            tcx.sess.bug(&format!("static call to invalid vtable: {}",
                                 vtbl.repr(tcx)));
        }
    }
}

fn method_with_name(ccx: &CrateContext, impl_id: ast::DefId, name: ast::Name)
                    -> ast::DefId {
    match ccx.impl_method_cache().borrow().get(&(impl_id, name)).cloned() {
        Some(m) => return m,
        None => {}
    }

    let impl_items = ccx.tcx().impl_items.borrow();
    let impl_items =
        impl_items.get(&impl_id)
                  .expect("could not find impl while translating");
    let meth_did = impl_items.iter()
                             .find(|&did| {
                                ty::impl_or_trait_item(ccx.tcx(), did.def_id()).name() == name
                             }).expect("could not find method while \
                                        translating");

    ccx.impl_method_cache().borrow_mut().insert((impl_id, name),
                                              meth_did.def_id());
    meth_did.def_id()
}

fn trans_monomorphized_callee<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                          method_call: MethodCall,
                                          trait_id: ast::DefId,
                                          n_method: uint,
                                          vtable: traits::Vtable<'tcx, ()>)
                                          -> Callee<'blk, 'tcx> {
    let _icx = push_ctxt("meth::trans_monomorphized_callee");
    match vtable {
        traits::VtableImpl(vtable_impl) => {
            let ccx = bcx.ccx();
            let impl_did = vtable_impl.impl_def_id;
            let mname = match ty::trait_item(ccx.tcx(), trait_id, n_method) {
                ty::MethodTraitItem(method) => method.name,
                ty::TypeTraitItem(_) => {
                    bcx.tcx().sess.bug("can't monomorphize an associated \
                                        type")
                }
            };
            let mth_id = method_with_name(bcx.ccx(), impl_did, mname);

            // create a concatenated set of substitutions which includes
            // those from the impl and those from the method:
            let callee_substs =
                combine_impl_and_methods_tps(
                    bcx, MethodCallKey(method_call), vtable_impl.substs);

            // translate the function
            let llfn = trans_fn_ref_with_substs(bcx.ccx(),
                                                mth_id,
                                                MethodCallKey(method_call),
                                                bcx.fcx.param_substs,
                                                callee_substs).val;

            Callee { bcx: bcx, data: Fn(llfn) }
        }
        traits::VtableClosure(closure_def_id, substs) => {
            // The substitutions should have no type parameters remaining
            // after passing through fulfill_obligation
            let llfn = trans_fn_ref_with_substs(bcx.ccx(),
                                                closure_def_id,
                                                MethodCallKey(method_call),
                                                bcx.fcx.param_substs,
                                                substs).val;

            Callee {
                bcx: bcx,
                data: Fn(llfn),
            }
        }
        traits::VtableFnPointer(fn_ty) => {
            let llfn = trans_fn_pointer_shim(bcx.ccx(), fn_ty);
            Callee { bcx: bcx, data: Fn(llfn) }
        }
        traits::VtableObject(ref data) => {
            let (llfn, _) = trans_object_shim(bcx.ccx(),
                                              data.object_ty,
                                              data.upcast_trait_ref.clone(),
                                              n_method);
            Callee { bcx: bcx, data: Fn(llfn) }
        }
        traits::VtableBuiltin(..) |
        traits::VtableDefaultImpl(..) |
        traits::VtableParam(..) => {
            bcx.sess().bug(
                &format!("resolved vtable bad vtable {} in trans",
                        vtable.repr(bcx.tcx())));
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

    debug!("rcvr_substs={}", rcvr_substs.repr(ccx.tcx()));
    debug!("node_substs={}", node_substs.repr(ccx.tcx()));

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
                                  method_ty: Ty<'tcx>,
                                  vtable_index: uint,
                                  self_expr: &ast::Expr,
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

    trans_trait_callee_from_llval(bcx, method_ty, vtable_index, llval)
}

/// Same as `trans_trait_callee()` above, except that it is given a by-ref pointer to the object
/// pair.
pub fn trans_trait_callee_from_llval<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                                 callee_ty: Ty<'tcx>,
                                                 vtable_index: uint,
                                                 llpair: ValueRef)
                                                 -> Callee<'blk, 'tcx> {
    let _icx = push_ctxt("meth::trans_trait_callee");
    let ccx = bcx.ccx();

    // Load the data pointer from the object.
    debug!("trans_trait_callee_from_llval(callee_ty={}, vtable_index={}, llpair={})",
           callee_ty.repr(ccx.tcx()),
           vtable_index,
           bcx.val_to_string(llpair));
    let llboxptr = GEPi(bcx, llpair, &[0, abi::FAT_PTR_ADDR]);
    let llbox = Load(bcx, llboxptr);
    let llself = PointerCast(bcx, llbox, Type::i8p(ccx));

    // Replace the self type (&Self or Box<Self>) with an opaque pointer.
    let llcallee_ty = match callee_ty.sty {
        ty::ty_bare_fn(_, ref f) if f.abi == Rust || f.abi == RustCall => {
            let fake_sig =
                ty::Binder(ty::FnSig {
                    inputs: f.sig.0.inputs[1..].to_vec(),
                    output: f.sig.0.output,
                    variadic: f.sig.0.variadic,
                });
            type_of_rust_fn(ccx, Some(Type::i8p(ccx)), &fake_sig, f.abi)
        }
        _ => {
            ccx.sess().bug("meth::trans_trait_callee given non-bare-rust-fn");
        }
    };
    let llvtable = Load(bcx,
                        PointerCast(bcx,
                                    GEPi(bcx, llpair,
                                         &[0, abi::FAT_PTR_EXTRA]),
                                    Type::vtable(ccx).ptr_to().ptr_to()));
    let mptr = Load(bcx, GEPi(bcx, llvtable, &[0, vtable_index + VTABLE_OFFSET]));
    let mptr = PointerCast(bcx, mptr, llcallee_ty.ptr_to());

    return Callee {
        bcx: bcx,
        data: TraitItem(MethodData {
            llfn: mptr,
            llself: llself,
        })
    };
}

/// Generate a shim function that allows an object type like `SomeTrait` to
/// implement the type `SomeTrait`. Imagine a trait definition:
///
///    trait SomeTrait { fn get(&self) -> int; ... }
///
/// And a generic bit of code:
///
///    fn foo<T:SomeTrait>(t: &T) {
///        let x = SomeTrait::get;
///        x(t)
///    }
///
/// What is the value of `x` when `foo` is invoked with `T=SomeTrait`?
/// The answer is that it it is a shim function generate by this
/// routine:
///
///    fn shim(t: &SomeTrait) -> int {
///        // ... call t.get() virtually ...
///    }
///
/// In fact, all virtual calls can be thought of as normal trait calls
/// that go through this shim function.
pub fn trans_object_shim<'a, 'tcx>(
    ccx: &'a CrateContext<'a, 'tcx>,
    object_ty: Ty<'tcx>,
    upcast_trait_ref: ty::PolyTraitRef<'tcx>,
    method_offset_in_trait: uint)
    -> (ValueRef, Ty<'tcx>)
{
    let _icx = push_ctxt("trans_object_shim");
    let tcx = ccx.tcx();
    let trait_id = upcast_trait_ref.def_id();

    debug!("trans_object_shim(object_ty={}, upcast_trait_ref={}, method_offset_in_trait={})",
           object_ty.repr(tcx),
           upcast_trait_ref.repr(tcx),
           method_offset_in_trait);

    let object_trait_ref =
        match object_ty.sty {
            ty::ty_trait(ref data) => {
                data.principal_trait_ref_with_self_ty(tcx, object_ty)
            }
            _ => {
                tcx.sess.bug(&format!("trans_object_shim() called on non-object: {}",
                                      object_ty.repr(tcx)));
            }
        };

    // Upcast to the trait in question and extract out the substitutions.
    let upcast_trait_ref = ty::erase_late_bound_regions(tcx, &upcast_trait_ref);
    let object_substs = upcast_trait_ref.substs.clone().erase_regions();
    debug!("trans_object_shim: object_substs={}", object_substs.repr(tcx));

    // Lookup the type of this method as declared in the trait and apply substitutions.
    let method_ty = match ty::trait_item(tcx, trait_id, method_offset_in_trait) {
        ty::MethodTraitItem(method) => method,
        ty::TypeTraitItem(_) => {
            tcx.sess.bug("can't create a method shim for an associated type")
        }
    };
    let fty = monomorphize::apply_param_substs(tcx, &object_substs, &method_ty.fty);
    let fty = tcx.mk_bare_fn(fty);
    let method_ty = opaque_method_ty(tcx, fty);
    debug!("trans_object_shim: fty={} method_ty={}", fty.repr(tcx), method_ty.repr(tcx));

    //
    let shim_fn_ty = ty::mk_bare_fn(tcx, None, fty);
    let method_bare_fn_ty = ty::mk_bare_fn(tcx, None, method_ty);
    let function_name =
        link::mangle_internal_name_by_type_and_seq(ccx, shim_fn_ty, "object_shim");
    let llfn =
        decl_internal_rust_fn(ccx, shim_fn_ty, &function_name);

    let sig = ty::erase_late_bound_regions(ccx.tcx(), &fty.sig);

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

    // the first argument (`self`) will be a trait object
    let llobject = get_param(fcx.llfn, fcx.arg_pos(0) as u32);

    debug!("trans_object_shim: llobject={}",
           bcx.val_to_string(llobject));

    // the remaining arguments will be, well, whatever they are
    let input_tys =
        match fty.abi {
            RustCall => {
                // unpack the tuple to extract the input type arguments:
                match sig.inputs[1].sty {
                    ty::ty_tup(ref tys) => &**tys,
                    _ => {
                        bcx.sess().bug(
                            &format!("rust-call expects a tuple not {}",
                                     sig.inputs[1].repr(tcx)));
                    }
                }
            }
            _ => {
                // skip the self parameter:
                &sig.inputs[1..]
            }
        };

    let llargs: Vec<_> =
        input_tys.iter()
        .enumerate()
        .map(|(i, _)| {
            let llarg = get_param(fcx.llfn, fcx.arg_pos(i+1) as u32);
            debug!("trans_object_shim: input #{} == {}",
                   i, bcx.val_to_string(llarg));
            llarg
        })
        .collect();

    assert!(!fcx.needs_ret_allocas);

    let sig =
        ty::erase_late_bound_regions(bcx.tcx(), &fty.sig);

    let dest =
        fcx.llretslotptr.get().map(
            |_| expr::SaveIn(fcx.get_ret_slot(bcx, sig.output, "ret_slot")));

    let method_offset_in_vtable =
        traits::get_vtable_index_of_object_method(bcx.tcx(),
                                                  object_trait_ref.clone(),
                                                  trait_id,
                                                  method_offset_in_trait);
    debug!("trans_object_shim: method_offset_in_vtable={}",
           method_offset_in_vtable);

    bcx = trans_call_inner(bcx,
                           DebugLoc::None,
                           method_bare_fn_ty,
                           |bcx, _| trans_trait_callee_from_llval(bcx,
                                                                  method_bare_fn_ty,
                                                                  method_offset_in_vtable,
                                                                  llobject),
                           ArgVals(&llargs),
                           dest).bcx;

    finish_fn(&fcx, bcx, sig.output, DebugLoc::None);

    (llfn, method_bare_fn_ty)
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

    debug!("get_vtable(trait_ref={})", trait_ref.repr(tcx));

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
            traits::VtableClosure(closure_def_id, substs) => {
                let llfn = trans_fn_ref_with_substs(
                    ccx,
                    closure_def_id,
                    ExprId(0),
                    param_substs,
                    substs).val;

                vec![llfn].into_iter()
            }
            traits::VtableFnPointer(bare_fn_ty) => {
                vec![trans_fn_pointer_shim(ccx, bare_fn_ty)].into_iter()
            }
            traits::VtableObject(ref data) => {
                // this would imply that the Self type being erased is
                // an object type; this cannot happen because we
                // cannot cast an unsized type into a trait object
                tcx.sess.bug(
                    &format!("cannot get vtable for an object type: {}",
                            data.repr(tcx)));
            }
            traits::VtableParam(..) => {
                tcx.sess.bug(
                    &format!("resolved vtable for {} to bad vtable {} in trans",
                            trait_ref.repr(tcx),
                            vtable.repr(tcx)));
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

    let vtable = consts::addr_of(ccx, C_struct(ccx, &components, false),
                                 "vtable", trait_ref.def_id().node);

    ccx.vtables().borrow_mut().insert(trait_ref, vtable);
    vtable
}

fn emit_vtable_methods<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                 impl_id: ast::DefId,
                                 substs: subst::Substs<'tcx>,
                                 param_substs: &'tcx subst::Substs<'tcx>)
                                 -> Vec<ValueRef>
{
    let tcx = ccx.tcx();

    debug!("emit_vtable_methods(impl_id={}, substs={}, param_substs={})",
           impl_id.repr(tcx),
           substs.repr(tcx),
           param_substs.repr(tcx));

    let trt_id = match ty::impl_trait_ref(tcx, impl_id) {
        Some(t_id) => t_id.def_id,
        None       => ccx.sess().bug("make_impl_vtable: don't know how to \
                                      make a vtable for a type impl!")
    };

    ty::populate_implementations_for_trait_if_necessary(tcx, trt_id);

    let trait_item_def_ids = ty::trait_item_def_ids(tcx, trt_id);
    trait_item_def_ids
        .iter()

        // Filter out the associated types.
        .filter_map(|item_def_id| {
            match *item_def_id {
                ty::MethodTraitItemId(def_id) => Some(def_id),
                ty::TypeTraitItemId(_) => None,
            }
        })

        // Now produce pointers for each remaining method. If the
        // method could never be called from this object, just supply
        // null.
        .map(|trait_method_def_id| {
            debug!("emit_vtable_methods: trait_method_def_id={}",
                   trait_method_def_id.repr(tcx));

            let trait_method_type = match ty::impl_or_trait_item(tcx, trait_method_def_id) {
                ty::MethodTraitItem(m) => m,
                ty::TypeTraitItem(_) => ccx.sess().bug("should be a method, not assoc type")
            };
            let name = trait_method_type.name;

            debug!("emit_vtable_methods: trait_method_type={}",
                   trait_method_type.repr(tcx));

            // The substitutions we have are on the impl, so we grab
            // the method type from the impl to substitute into.
            let impl_method_def_id = method_with_name(ccx, impl_id, name);
            let impl_method_type = match ty::impl_or_trait_item(tcx, impl_method_def_id) {
                ty::MethodTraitItem(m) => m,
                ty::TypeTraitItem(_) => ccx.sess().bug("should be a method, not assoc type")
            };

            debug!("emit_vtable_methods: m={}",
                   impl_method_type.repr(tcx));

            let nullptr = C_null(Type::nil(ccx).ptr_to());

            if impl_method_type.generics.has_type_params(subst::FnSpace) {
                debug!("emit_vtable_methods: generic");
                return nullptr;
            }

            let bare_fn_ty =
                ty::mk_bare_fn(tcx, None, tcx.mk_bare_fn(impl_method_type.fty.clone()));
            if ty::type_has_self(bare_fn_ty) {
                debug!("emit_vtable_methods: type_has_self {}",
                       bare_fn_ty.repr(tcx));
                return nullptr;
            }

            // If this is a default method, it's possible that it
            // relies on where clauses that do not hold for this
            // particular set of type parameters. Note that this
            // method could then never be called, so we do not want to
            // try and trans it, in that case. Issue #23435.
            if ty::provided_source(tcx, impl_method_def_id).is_some() {
                let predicates =
                    monomorphize::apply_param_substs(tcx,
                                                     &substs,
                                                     &impl_method_type.predicates.predicates);
                if !predicates_hold(ccx, predicates.into_vec()) {
                    debug!("emit_vtable_methods: predicates do not hold");
                    return nullptr;
                }
            }

            trans_fn_ref_with_substs(ccx,
                                     impl_method_def_id,
                                     ExprId(0),
                                     param_substs,
                                     substs.clone()).val
        })
        .collect()
}

/// Replace the self type (&Self or Box<Self>) with an opaque pointer.
pub fn opaque_method_ty<'tcx>(tcx: &ty::ctxt<'tcx>, method_ty: &ty::BareFnTy<'tcx>)
        -> &'tcx ty::BareFnTy<'tcx> {
    let mut inputs = method_ty.sig.0.inputs.clone();
    inputs[0] = ty::mk_mut_ptr(tcx, ty::mk_mach_int(tcx, ast::TyI8));

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
