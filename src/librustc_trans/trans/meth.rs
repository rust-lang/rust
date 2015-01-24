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
use llvm::{self, ValueRef, get_param};
use metadata::csearch;
use middle::subst::{Subst, Substs};
use middle::subst::VecPerParamSpace;
use middle::subst;
use middle::traits;
use trans::base::*;
use trans::build::*;
use trans::callee::*;
use trans::callee;
use trans::cleanup;
use trans::common::*;
use trans::datum::*;
use trans::debuginfo::DebugLoc;
use trans::expr::{SaveIn, Ignore};
use trans::expr;
use trans::glue;
use trans::machine;
use trans::type_::Type;
use trans::type_of::*;
use middle::ty::{self, Ty};
use middle::ty::MethodCall;
use util::ppaux::Repr;

use std::ffi::CString;
use std::rc::Rc;
use syntax::abi::{Rust, RustCall};
use syntax::parse::token;
use syntax::{ast, ast_map, attr, visit};
use syntax::ast_util::PostExpansionMethod;
use syntax::codemap::DUMMY_SP;

// drop_glue pointer, size, align.
static VTABLE_OFFSET: uint = 3;

/// The main "translation" pass for methods.  Generates code
/// for non-monomorphized methods only.  Other methods will
/// be generated once they are invoked with specific type parameters,
/// see `trans::base::lval_static_fn()` or `trans::base::monomorphic_fn()`.
pub fn trans_impl(ccx: &CrateContext,
                  name: ast::Ident,
                  impl_items: &[ast::ImplItem],
                  generics: &ast::Generics,
                  id: ast::NodeId) {
    let _icx = push_ctxt("meth::trans_impl");
    let tcx = ccx.tcx();

    debug!("trans_impl(name={}, id={})", name.repr(tcx), id);

    // Both here and below with generic methods, be sure to recurse and look for
    // items that we need to translate.
    if !generics.ty_params.is_empty() {
        let mut v = TransItemVisitor{ ccx: ccx };
        for impl_item in impl_items.iter() {
            match *impl_item {
                ast::MethodImplItem(ref method) => {
                    visit::walk_method_helper(&mut v, &**method);
                }
                ast::TypeImplItem(_) => {}
            }
        }
        return;
    }
    for impl_item in impl_items.iter() {
        match *impl_item {
            ast::MethodImplItem(ref method) => {
                if method.pe_generics().ty_params.len() == 0u {
                    let trans_everywhere = attr::requests_inline(&method.attrs[]);
                    for (ref ccx, is_origin) in ccx.maybe_iter(trans_everywhere) {
                        let llfn = get_item_val(ccx, method.id);
                        trans_fn(ccx,
                                 method.pe_fn_decl(),
                                 method.pe_body(),
                                 llfn,
                                 &Substs::trans_empty(),
                                 method.id,
                                 &[]);
                        update_linkage(ccx,
                                       llfn,
                                       Some(method.id),
                                       if is_origin { OriginalTranslation } else { InlinedCopy });
                    }
                }
                let mut v = TransItemVisitor {
                    ccx: ccx,
                };
                visit::walk_method_helper(&mut v, &**method);
            }
            ast::TypeImplItem(_) => {}
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
        ty::MethodStaticUnboxedClosure(did) => {
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
            method_num
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
                               mt.real_index,
                               self_expr,
                               arg_cleanup_scope)
        }
    }
}

pub fn trans_static_method_callee<'a, 'tcx>(ccx: &CrateContext<'a, 'tcx>,
                                            method_id: ast::DefId,
                                            trait_id: ast::DefId,
                                            expr_id: ast::NodeId,
                                            param_substs: &subst::Substs<'tcx>)
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
            ast_map::NodeTraitItem(method) => {
                let ident = match *method {
                    ast::RequiredMethod(ref m) => m.ident,
                    ast::ProvidedMethod(ref m) => m.pe_ident(),
                    ast::TypeTraitItem(_) => {
                        tcx.sess.bug("trans_static_method_callee() on \
                                      an associated type?!")
                    }
                };
                ident.name
            }
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
                trans_object_shim(ccx, data.object_ty, trait_id, method_offset_in_trait);
            immediate_rvalue(llfn, ty)
        }
        _ => {
            tcx.sess.bug(&format!("static call to invalid vtable: {}",
                                 vtbl.repr(tcx))[]);
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
        traits::VtableUnboxedClosure(closure_def_id, substs) => {
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
            let (llfn, _) = trans_object_shim(bcx.ccx(), data.object_ty, trait_id, n_method);
            Callee { bcx: bcx, data: Fn(llfn) }
        }
        traits::VtableBuiltin(..) |
        traits::VtableParam(..) => {
            bcx.sess().bug(
                &format!("resolved vtable bad vtable {} in trans",
                        vtable.repr(bcx.tcx()))[]);
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
                                  n_method: uint,
                                  self_expr: &ast::Expr,
                                  arg_cleanup_scope: cleanup::ScopeId)
                                  -> Callee<'blk, 'tcx> {
    let _icx = push_ctxt("meth::trans_trait_callee");
    let mut bcx = bcx;

    // Translate self_datum and take ownership of the value by
    // converting to an rvalue.
    let self_datum = unpack_datum!(
        bcx, expr::trans(bcx, self_expr));

    let llval = if type_needs_drop(bcx.tcx(), self_datum.ty) {
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

    trans_trait_callee_from_llval(bcx, method_ty, n_method, llval)
}

/// Same as `trans_trait_callee()` above, except that it is given a by-ref pointer to the object
/// pair.
pub fn trans_trait_callee_from_llval<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                                 callee_ty: Ty<'tcx>,
                                                 n_method: uint,
                                                 llpair: ValueRef)
                                                 -> Callee<'blk, 'tcx> {
    let _icx = push_ctxt("meth::trans_trait_callee");
    let ccx = bcx.ccx();

    // Load the data pointer from the object.
    debug!("(translating trait callee) loading second index from pair");
    let llboxptr = GEPi(bcx, llpair, &[0u, abi::FAT_PTR_ADDR]);
    let llbox = Load(bcx, llboxptr);
    let llself = PointerCast(bcx, llbox, Type::i8p(ccx));

    // Load the function from the vtable and cast it to the expected type.
    debug!("(translating trait callee) loading method");

    // Replace the self type (&Self or Box<Self>) with an opaque pointer.
    let llcallee_ty = match callee_ty.sty {
        ty::ty_bare_fn(_, ref f) if f.abi == Rust || f.abi == RustCall => {
            let fake_sig =
                ty::Binder(ty::FnSig {
                    inputs: f.sig.0.inputs[1..].to_vec(),
                    output: f.sig.0.output,
                    variadic: f.sig.0.variadic,
                });
            type_of_rust_fn(ccx,
                            Some(Type::i8p(ccx)),
                            &fake_sig,
                            f.abi)
        }
        _ => {
            ccx.sess().bug("meth::trans_trait_callee given non-bare-rust-fn");
        }
    };
    let llvtable = Load(bcx,
                        PointerCast(bcx,
                                    GEPi(bcx, llpair,
                                         &[0u, abi::FAT_PTR_EXTRA]),
                                    Type::vtable(ccx).ptr_to().ptr_to()));
    let mptr = Load(bcx, GEPi(bcx, llvtable, &[0u, n_method + VTABLE_OFFSET]));
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
    trait_id: ast::DefId,
    method_offset_in_trait: uint)
    -> (ValueRef, Ty<'tcx>)
{
    let _icx = push_ctxt("trans_object_shim");
    let tcx = ccx.tcx();

    debug!("trans_object_shim(object_ty={}, trait_id={}, n_method={})",
           object_ty.repr(tcx),
           trait_id.repr(tcx),
           method_offset_in_trait);

    let object_trait_ref =
        match object_ty.sty {
            ty::ty_trait(ref data) => {
                data.principal_trait_ref_with_self_ty(tcx, object_ty)
            }
            _ => {
                tcx.sess.bug(format!("trans_object_shim() called on non-object: {}",
                                     object_ty.repr(tcx)).as_slice());
            }
        };

    // Upcast to the trait in question and extract out the substitutions.
    let upcast_trait_ref = traits::upcast(ccx.tcx(), object_trait_ref.clone(), trait_id).unwrap();
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
    let fty = method_ty.fty.subst(tcx, &object_substs);
    let fty = tcx.mk_bare_fn(fty);
    debug!("trans_object_shim: fty={}", fty.repr(tcx));

    //
    let method_bare_fn_ty =
        ty::mk_bare_fn(tcx, None, fty);
    let function_name =
        link::mangle_internal_name_by_type_and_seq(ccx, method_bare_fn_ty, "object_shim");
    let llfn =
        decl_internal_rust_fn(ccx, method_bare_fn_ty, function_name.as_slice());

    let sig = ty::erase_late_bound_regions(ccx.tcx(), &fty.sig);

    //
    let block_arena = TypedArena::new();
    let empty_substs = Substs::trans_empty();
    let fcx = new_fn_ctxt(ccx,
                          llfn,
                          ast::DUMMY_NODE_ID,
                          false,
                          sig.output,
                          &empty_substs,
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
                    ty::ty_tup(ref tys) => tys.as_slice(),
                    _ => {
                        bcx.sess().bug(
                            format!("rust-call expects a tuple not {}",
                                    sig.inputs[1].repr(tcx)).as_slice());
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
                           None,
                           method_bare_fn_ty,
                           |bcx, _| trans_trait_callee_from_llval(bcx,
                                                                  method_bare_fn_ty,
                                                                  method_offset_in_vtable,
                                                                  llobject),
                           ArgVals(llargs.as_slice()),
                           dest).bcx;

    finish_fn(&fcx, bcx, sig.output, DebugLoc::None);

    (llfn, method_bare_fn_ty)
}

/// Creates a returns a dynamic vtable for the given type and vtable origin.
/// This is used only for objects.
///
/// The `trait_ref` encodes the erased self type. Hence if we are
/// making an object `Foo<Trait>` from a value of type `Foo<T>`, then
/// `trait_ref` would map `T:Trait`, but `box_ty` would be
/// `Foo<T>`. This `box_ty` is primarily used to encode the destructor.
/// This will hopefully change now that DST is underway.
pub fn get_vtable<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                              box_ty: Ty<'tcx>,
                              trait_ref: ty::PolyTraitRef<'tcx>)
                              -> ValueRef
{
    debug!("get_vtable(box_ty={}, trait_ref={})",
           box_ty.repr(bcx.tcx()),
           trait_ref.repr(bcx.tcx()));

    let tcx = bcx.tcx();
    let ccx = bcx.ccx();
    let _icx = push_ctxt("meth::get_vtable");

    // Check the cache.
    let cache_key = (box_ty, trait_ref.clone());
    match ccx.vtables().borrow().get(&cache_key) {
        Some(&val) => { return val }
        None => { }
    }

    // Not in the cache. Build it.
    let methods = traits::supertraits(tcx, trait_ref.clone()).flat_map(|trait_ref| {
        let vtable = fulfill_obligation(bcx.ccx(),
                                        DUMMY_SP,
                                        trait_ref.clone());
        match vtable {
            traits::VtableBuiltin(_) => {
                Vec::new().into_iter()
            }
            traits::VtableImpl(
                traits::VtableImplData {
                    impl_def_id: id,
                    substs,
                    nested: _ }) => {
                emit_vtable_methods(bcx, id, substs).into_iter()
            }
            traits::VtableUnboxedClosure(closure_def_id, substs) => {
                let llfn = trans_fn_ref_with_substs(
                    bcx.ccx(),
                    closure_def_id,
                    ExprId(0),
                    bcx.fcx.param_substs,
                    substs.clone()).val;

                (vec!(llfn)).into_iter()
            }
            traits::VtableFnPointer(bare_fn_ty) => {
                let llfn = vec![trans_fn_pointer_shim(bcx.ccx(), bare_fn_ty)];
                llfn.into_iter()
            }
            traits::VtableObject(ref data) => {
                // this would imply that the Self type being erased is
                // an object type; this cannot happen because we
                // cannot cast an unsized type into a trait object
                bcx.sess().bug(
                    format!("cannot get vtable for an object type: {}",
                            data.repr(bcx.tcx())).as_slice());
            }
            traits::VtableParam(..) => {
                bcx.sess().bug(
                    &format!("resolved vtable for {} to bad vtable {} in trans",
                            trait_ref.repr(bcx.tcx()),
                            vtable.repr(bcx.tcx()))[]);
            }
        }
    });

    let size_ty = sizing_type_of(ccx, trait_ref.self_ty());
    let size = machine::llsize_of_alloc(ccx, size_ty);
    let ll_size = C_uint(ccx, size);
    let align = align_of(ccx, trait_ref.self_ty());
    let ll_align = C_uint(ccx, align);

    // Generate a destructor for the vtable.
    let drop_glue = glue::get_drop_glue(ccx, box_ty);
    let vtable = make_vtable(ccx, drop_glue, ll_size, ll_align, methods);

    ccx.vtables().borrow_mut().insert(cache_key, vtable);
    vtable
}

/// Helper function to declare and initialize the vtable.
pub fn make_vtable<I: Iterator<Item=ValueRef>>(ccx: &CrateContext,
                                          drop_glue: ValueRef,
                                          size: ValueRef,
                                          align: ValueRef,
                                          ptrs: I)
                                          -> ValueRef {
    let _icx = push_ctxt("meth::make_vtable");

    let head = vec![drop_glue, size, align];
    let components: Vec<_> = head.into_iter().chain(ptrs).collect();

    unsafe {
        let tbl = C_struct(ccx, &components[], false);
        let sym = token::gensym("vtable");
        let buf = CString::from_vec(format!("vtable{}", sym.usize()).into_bytes());
        let vt_gvar = llvm::LLVMAddGlobal(ccx.llmod(), val_ty(tbl).to_ref(),
                                          buf.as_ptr());
        llvm::LLVMSetInitializer(vt_gvar, tbl);
        llvm::LLVMSetGlobalConstant(vt_gvar, llvm::True);
        llvm::SetLinkage(vt_gvar, llvm::InternalLinkage);
        vt_gvar
    }
}

fn emit_vtable_methods<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                   impl_id: ast::DefId,
                                   substs: subst::Substs<'tcx>)
                                   -> Vec<ValueRef> {
    let ccx = bcx.ccx();
    let tcx = ccx.tcx();

    let trt_id = match ty::impl_trait_ref(tcx, impl_id) {
        Some(t_id) => t_id.def_id,
        None       => ccx.sess().bug("make_impl_vtable: don't know how to \
                                      make a vtable for a type impl!")
    };

    ty::populate_implementations_for_trait_if_necessary(bcx.tcx(), trt_id);

    let trait_item_def_ids = ty::trait_item_def_ids(tcx, trt_id);
    trait_item_def_ids.iter().flat_map(|method_def_id| {
        let method_def_id = method_def_id.def_id();
        let name = ty::impl_or_trait_item(tcx, method_def_id).name();
        // The substitutions we have are on the impl, so we grab
        // the method type from the impl to substitute into.
        let m_id = method_with_name(ccx, impl_id, name);
        let ti = ty::impl_or_trait_item(tcx, m_id);
        match ti {
            ty::MethodTraitItem(m) => {
                debug!("(making impl vtable) emitting method {} at subst {}",
                       m.repr(tcx),
                       substs.repr(tcx));
                if m.generics.has_type_params(subst::FnSpace) ||
                    ty::type_has_self(ty::mk_bare_fn(tcx, None, tcx.mk_bare_fn(m.fty.clone())))
                {
                    debug!("(making impl vtable) method has self or type \
                            params: {}",
                           token::get_name(name));
                    Some(C_null(Type::nil(ccx).ptr_to())).into_iter()
                } else {
                    let fn_ref = trans_fn_ref_with_substs(
                        ccx,
                        m_id,
                        ExprId(0),
                        bcx.fcx.param_substs,
                        substs.clone()).val;

                    // currently, at least, by-value self is not object safe
                    assert!(m.explicit_self != ty::ByValueExplicitSelfCategory);

                    Some(fn_ref).into_iter()
                }
            }
            ty::TypeTraitItem(_) => {
                None.into_iter()
            }
        }
    }).collect()
}

/// Generates the code to convert from a pointer (`Box<T>`, `&T`, etc) into an object
/// (`Box<Trait>`, `&Trait`, etc). This means creating a pair where the first word is the vtable
/// and the second word is the pointer.
pub fn trans_trait_cast<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                    datum: Datum<'tcx, Expr>,
                                    id: ast::NodeId,
                                    trait_ref: ty::PolyTraitRef<'tcx>,
                                    dest: expr::Dest)
                                    -> Block<'blk, 'tcx> {
    let mut bcx = bcx;
    let _icx = push_ctxt("meth::trans_trait_cast");

    let lldest = match dest {
        Ignore => {
            return datum.clean(bcx, "trait_trait_cast", id);
        }
        SaveIn(dest) => dest
    };

    debug!("trans_trait_cast: trait_ref={}",
           trait_ref.repr(bcx.tcx()));

    let datum_ty = datum.ty;
    let llbox_ty = type_of(bcx.ccx(), datum_ty);

    // Store the pointer into the first half of pair.
    let llboxdest = GEPi(bcx, lldest, &[0u, abi::FAT_PTR_ADDR]);
    let llboxdest = PointerCast(bcx, llboxdest, llbox_ty.ptr_to());
    bcx = datum.store_to(bcx, llboxdest);

    // Store the vtable into the second half of pair.
    let vtable = get_vtable(bcx, datum_ty, trait_ref);
    let llvtabledest = GEPi(bcx, lldest, &[0u, abi::FAT_PTR_EXTRA]);
    let llvtabledest = PointerCast(bcx, llvtabledest, val_ty(vtable).ptr_to());
    Store(bcx, vtable, llvtabledest);

    bcx
}
