// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use back::abi;
use llvm;
use llvm::ValueRef;
use metadata::csearch;
use middle::subst::{Subst,Substs};
use middle::subst::VecPerParamSpace;
use middle::subst;
use middle::traits;
use middle::trans::base::*;
use middle::trans::build::*;
use middle::trans::callee::*;
use middle::trans::callee;
use middle::trans::cleanup;
use middle::trans::common::*;
use middle::trans::datum::*;
use middle::trans::expr::{SaveIn, Ignore};
use middle::trans::expr;
use middle::trans::glue;
use middle::trans::machine;
use middle::trans::type_::Type;
use middle::trans::type_of::*;
use middle::ty;
use middle::typeck;
use middle::typeck::MethodCall;
use util::ppaux::Repr;

use std::c_str::ToCStr;
use std::rc::Rc;
use syntax::abi::{Rust, RustCall};
use syntax::parse::token;
use syntax::{ast, ast_map, attr, visit};
use syntax::ast_util::PostExpansionMethod;
use syntax::codemap::DUMMY_SP;

// drop_glue pointer, size, align.
static VTABLE_OFFSET: uint = 3;

/**
The main "translation" pass for methods.  Generates code
for non-monomorphized methods only.  Other methods will
be generated once they are invoked with specific type parameters,
see `trans::base::lval_static_fn()` or `trans::base::monomorphic_fn()`.
*/
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
                    let trans_everywhere = attr::requests_inline(method.attrs.as_slice());
                    for (ref ccx, is_origin) in ccx.maybe_iter(trans_everywhere) {
                        let llfn = get_item_val(ccx, method.id);
                        trans_fn(ccx,
                                 method.pe_fn_decl(),
                                 method.pe_body(),
                                 llfn,
                                 &param_substs::empty(),
                                 method.id,
                                 []);
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
                 .find(&method_call)
                 .map(|method| (method.origin.clone(), method.ty))
                 .unwrap();

    match origin {
        typeck::MethodStatic(did) |
        typeck::MethodStaticUnboxedClosure(did) => {
            Callee {
                bcx: bcx,
                data: Fn(callee::trans_fn_ref(bcx,
                                              did,
                                              MethodCall(method_call))),
            }
        }

        typeck::MethodTypeParam(typeck::MethodParam {
            trait_ref: ref trait_ref,
            method_num: method_num
        }) => {
            let trait_ref =
                Rc::new(trait_ref.subst(bcx.tcx(),
                                        &bcx.fcx.param_substs.substs));
            let span = bcx.tcx().map.span(method_call.expr_id);
            let origin = fulfill_obligation(bcx.ccx(),
                                            span,
                                            (*trait_ref).clone());
            debug!("origin = {}", origin.repr(bcx.tcx()));
            trans_monomorphized_callee(bcx, method_call, trait_ref.def_id,
                                       method_num, origin)
        }

        typeck::MethodTraitObject(ref mt) => {
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

pub fn trans_static_method_callee(bcx: Block,
                                  method_id: ast::DefId,
                                  trait_id: ast::DefId,
                                  expr_id: ast::NodeId)
                                  -> ValueRef
{
    let _icx = push_ctxt("meth::trans_static_method_callee");
    let ccx = bcx.ccx();

    debug!("trans_static_method_callee(method_id={}, trait_id={}, \
            expr_id={})",
           method_id,
           ty::item_path_str(bcx.tcx(), trait_id),
           expr_id);

    let mname = if method_id.krate == ast::LOCAL_CRATE {
        match bcx.tcx().map.get(method_id.node) {
            ast_map::NodeTraitItem(method) => {
                let ident = match *method {
                    ast::RequiredMethod(ref m) => m.ident,
                    ast::ProvidedMethod(ref m) => m.pe_ident(),
                    ast::TypeTraitItem(_) => {
                        bcx.tcx().sess.bug("trans_static_method_callee() on \
                                            an associated type?!")
                    }
                };
                ident.name
            }
            _ => fail!("callee is not a trait method")
        }
    } else {
        csearch::get_item_path(bcx.tcx(), method_id).last().unwrap().name()
    };
    debug!("trans_static_method_callee: method_id={}, expr_id={}, \
            name={}", method_id, expr_id, token::get_name(mname));

    // Find the substitutions for the fn itself. This includes
    // type parameters that belong to the trait but also some that
    // belong to the method:
    let rcvr_substs = node_id_substs(bcx, ExprId(expr_id));
    let (rcvr_type, rcvr_self, rcvr_method) = rcvr_substs.types.split();

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
    debug!("trait_substs={}", trait_substs.repr(bcx.tcx()));
    let trait_ref = Rc::new(ty::TraitRef { def_id: trait_id,
                                           substs: trait_substs });
    let vtbl = fulfill_obligation(bcx.ccx(),
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
            let (impl_type, impl_self, _) = impl_substs.types.split();
            let callee_substs =
                Substs::erased(VecPerParamSpace::new(impl_type,
                                                     impl_self,
                                                     rcvr_method));

            let mth_id = method_with_name(ccx, impl_did, mname);
            let llfn = trans_fn_ref_with_substs(bcx, mth_id, ExprId(expr_id),
                                                callee_substs);

            let callee_ty = node_id_type(bcx, expr_id);
            let llty = type_of_fn_from_ty(ccx, callee_ty).ptr_to();
            PointerCast(bcx, llfn, llty)
        }
        _ => {
            bcx.tcx().sess.bug(
                format!("static call to invalid vtable: {}",
                        vtbl.repr(bcx.tcx())).as_slice());
        }
    }
}

fn method_with_name(ccx: &CrateContext, impl_id: ast::DefId, name: ast::Name)
                    -> ast::DefId {
    match ccx.impl_method_cache().borrow().find_copy(&(impl_id, name)) {
        Some(m) => return m,
        None => {}
    }

    let impl_items = ccx.tcx().impl_items.borrow();
    let impl_items =
        impl_items.find(&impl_id)
                  .expect("could not find impl while translating");
    let meth_did = impl_items.iter()
                             .find(|&did| {
                                ty::impl_or_trait_item(ccx.tcx(),
                                                       did.def_id()).ident()
                                                                    .name ==
                                    name
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
                                          vtable: traits::Vtable<()>)
                                          -> Callee<'blk, 'tcx> {
    let _icx = push_ctxt("meth::trans_monomorphized_callee");
    match vtable {
        traits::VtableImpl(vtable_impl) => {
            let ccx = bcx.ccx();
            let impl_did = vtable_impl.impl_def_id;
            let mname = match ty::trait_item(ccx.tcx(), trait_id, n_method) {
                ty::MethodTraitItem(method) => method.ident,
                ty::TypeTraitItem(_) => {
                    bcx.tcx().sess.bug("can't monomorphize an associated \
                                        type")
                }
            };
            let mth_id = method_with_name(bcx.ccx(), impl_did, mname.name);

            // create a concatenated set of substitutions which includes
            // those from the impl and those from the method:
            let callee_substs =
                combine_impl_and_methods_tps(
                    bcx, MethodCall(method_call), vtable_impl.substs);

            // translate the function
            let llfn = trans_fn_ref_with_substs(bcx,
                                                mth_id,
                                                MethodCall(method_call),
                                                callee_substs);

            Callee { bcx: bcx, data: Fn(llfn) }
        }
        traits::VtableUnboxedClosure(closure_def_id) => {
          // The static region and type parameters are lies, but we're in
          // trans so it doesn't matter.
          //
          // FIXME(pcwalton): Is this true in the case of type parameters?
          let callee_substs = get_callee_substitutions_for_unboxed_closure(
                bcx,
                closure_def_id);

            let llfn = trans_fn_ref_with_substs(bcx,
                                                closure_def_id,
                                                MethodCall(method_call),
                                                callee_substs);

            Callee {
                bcx: bcx,
                data: Fn(llfn),
            }
        }
        _ => {
            bcx.tcx().sess.bug(
                "vtable_param left in monomorphized function's vtable substs");
        }
    }
}

fn combine_impl_and_methods_tps(bcx: Block,
                                node: ExprOrMethodCall,
                                rcvr_substs: subst::Substs)
                                -> subst::Substs
{
    /*!
     * Creates a concatenated set of substitutions which includes
     * those from the impl and those from the method.  This are
     * some subtle complications here.  Statically, we have a list
     * of type parameters like `[T0, T1, T2, M1, M2, M3]` where
     * `Tn` are type parameters that appear on the receiver.  For
     * example, if the receiver is a method parameter `A` with a
     * bound like `trait<B,C,D>` then `Tn` would be `[B,C,D]`.
     *
     * The weird part is that the type `A` might now be bound to
     * any other type, such as `foo<X>`.  In that case, the vector
     * we want is: `[X, M1, M2, M3]`.  Therefore, what we do now is
     * to slice off the method type parameters and append them to
     * the type parameters from the type that the receiver is
     * mapped to.
     */

    let ccx = bcx.ccx();

    let node_substs = node_id_substs(bcx, node);

    debug!("rcvr_substs={}", rcvr_substs.repr(ccx.tcx()));
    debug!("node_substs={}", node_substs.repr(ccx.tcx()));

    // Break apart the type parameters from the node and type
    // parameters from the receiver.
    let (_, _, node_method) = node_substs.types.split();
    let (rcvr_type, rcvr_self, rcvr_method) = rcvr_substs.types.clone().split();
    assert!(rcvr_method.is_empty());
    subst::Substs {
        regions: subst::ErasedRegions,
        types: subst::VecPerParamSpace::new(rcvr_type, rcvr_self, node_method)
    }
}

fn trans_trait_callee<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                  method_ty: ty::t,
                                  n_method: uint,
                                  self_expr: &ast::Expr,
                                  arg_cleanup_scope: cleanup::ScopeId)
                                  -> Callee<'blk, 'tcx> {
    /*!
     * Create a method callee where the method is coming from a trait
     * object (e.g., Box<Trait> type).  In this case, we must pull the fn
     * pointer out of the vtable that is packaged up with the object.
     * Objects are represented as a pair, so we first evaluate the self
     * expression and then extract the self data and vtable out of the
     * pair.
     */

    let _icx = push_ctxt("meth::trans_trait_callee");
    let mut bcx = bcx;

    // Translate self_datum and take ownership of the value by
    // converting to an rvalue.
    let self_datum = unpack_datum!(
        bcx, expr::trans(bcx, self_expr));

    let llval = if ty::type_needs_drop(bcx.tcx(), self_datum.ty) {
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

pub fn trans_trait_callee_from_llval<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                                 callee_ty: ty::t,
                                                 n_method: uint,
                                                 llpair: ValueRef)
                                                 -> Callee<'blk, 'tcx> {
    /*!
     * Same as `trans_trait_callee()` above, except that it is given
     * a by-ref pointer to the object pair.
     */

    let _icx = push_ctxt("meth::trans_trait_callee");
    let ccx = bcx.ccx();

    // Load the data pointer from the object.
    debug!("(translating trait callee) loading second index from pair");
    let llboxptr = GEPi(bcx, llpair, [0u, abi::trt_field_box]);
    let llbox = Load(bcx, llboxptr);
    let llself = PointerCast(bcx, llbox, Type::i8p(ccx));

    // Load the function from the vtable and cast it to the expected type.
    debug!("(translating trait callee) loading method");
    // Replace the self type (&Self or Box<Self>) with an opaque pointer.
    let llcallee_ty = match ty::get(callee_ty).sty {
        ty::ty_bare_fn(ref f) if f.abi == Rust || f.abi == RustCall => {
            type_of_rust_fn(ccx,
                            Some(Type::i8p(ccx)),
                            f.sig.inputs.slice_from(1),
                            f.sig.output,
                            f.abi)
        }
        _ => {
            ccx.sess().bug("meth::trans_trait_callee given non-bare-rust-fn");
        }
    };
    let llvtable = Load(bcx,
                        PointerCast(bcx,
                                    GEPi(bcx, llpair,
                                         [0u, abi::trt_field_vtable]),
                                    Type::vtable(ccx).ptr_to().ptr_to()));
    let mptr = Load(bcx, GEPi(bcx, llvtable, [0u, n_method + VTABLE_OFFSET]));
    let mptr = PointerCast(bcx, mptr, llcallee_ty.ptr_to());

    return Callee {
        bcx: bcx,
        data: TraitItem(MethodData {
            llfn: mptr,
            llself: llself,
        })
    };
}

/// Creates the self type and (fake) callee substitutions for an unboxed
/// closure with the given def ID. The static region and type parameters are
/// lies, but we're in trans so it doesn't matter.
fn get_callee_substitutions_for_unboxed_closure(bcx: Block,
                                                def_id: ast::DefId)
                                                -> subst::Substs {
    let self_ty = ty::mk_unboxed_closure(bcx.tcx(), def_id, ty::ReStatic);
    subst::Substs::erased(
        VecPerParamSpace::new(Vec::new(),
                              vec![
                                  ty::mk_rptr(bcx.tcx(),
                                              ty::ReStatic,
                                              ty::mt {
                                                ty: self_ty,
                                                mutbl: ast::MutMutable,
                                              })
                              ],
                              Vec::new()))
}

/// Creates a returns a dynamic vtable for the given type and vtable origin.
/// This is used only for objects.
///
/// The `trait_ref` encodes the erased self type. Hence if we are
/// making an object `Foo<Trait>` from a value of type `Foo<T>`, then
/// `trait_ref` would map `T:Trait`, but `box_ty` would be
/// `Foo<T>`. This `box_ty` is primarily used to encode the destructor.
/// This will hopefully change now that DST is underway.
pub fn get_vtable(bcx: Block,
                  box_ty: ty::t,
                  trait_ref: Rc<ty::TraitRef>)
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
    match ccx.vtables().borrow().find(&cache_key) {
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
                    substs: substs,
                    nested: _ }) => {
                emit_vtable_methods(bcx, id, substs).into_iter()
            }
            traits::VtableUnboxedClosure(closure_def_id) => {
                let callee_substs =
                    get_callee_substitutions_for_unboxed_closure(
                        bcx,
                        closure_def_id);

                let mut llfn = trans_fn_ref_with_substs(
                    bcx,
                    closure_def_id,
                    ExprId(0),
                    callee_substs.clone());

                {
                    let unboxed_closures = bcx.tcx()
                                              .unboxed_closures
                                              .borrow();
                    let closure_info =
                        unboxed_closures.find(&closure_def_id)
                                        .expect("get_vtable(): didn't find \
                                                 unboxed closure");
                    if closure_info.kind == ty::FnOnceUnboxedClosureKind {
                        // Untuple the arguments and create an unboxing shim.
                        let mut new_inputs = vec![
                            ty::mk_unboxed_closure(bcx.tcx(),
                                                   closure_def_id,
                                                   ty::ReStatic)
                        ];
                        match ty::get(closure_info.closure_type
                                                  .sig
                                                  .inputs[0]).sty {
                            ty::ty_tup(ref elements) => {
                                for element in elements.iter() {
                                    new_inputs.push(*element)
                                }
                            }
                            ty::ty_nil => {}
                            _ => {
                                bcx.tcx().sess.bug("get_vtable(): closure \
                                                    type wasn't a tuple")
                            }
                        }

                        let closure_type = ty::BareFnTy {
                            fn_style: closure_info.closure_type.fn_style,
                            abi: Rust,
                            sig: ty::FnSig {
                                binder_id: closure_info.closure_type
                                                       .sig
                                                       .binder_id,
                                inputs: new_inputs,
                                output: closure_info.closure_type.sig.output,
                                variadic: false,
                            },
                        };
                        debug!("get_vtable(): closure type is {}",
                               closure_type.repr(bcx.tcx()));
                        llfn = trans_unboxing_shim(bcx,
                                                   llfn,
                                                   &closure_type,
                                                   closure_def_id,
                                                   callee_substs);
                    }
                }

                (vec!(llfn)).into_iter()
            }
            traits::VtableParam(..) => {
                bcx.sess().bug(
                    format!("resolved vtable for {} to bad vtable {} in trans",
                            trait_ref.repr(bcx.tcx()),
                            vtable.repr(bcx.tcx())).as_slice());
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
pub fn make_vtable<I: Iterator<ValueRef>>(ccx: &CrateContext,
                                          drop_glue: ValueRef,
                                          size: ValueRef,
                                          align: ValueRef,
                                          ptrs: I)
                                          -> ValueRef {
    let _icx = push_ctxt("meth::make_vtable");

    let head = vec![drop_glue, size, align];
    let components: Vec<_> = head.into_iter().chain(ptrs).collect();

    unsafe {
        let tbl = C_struct(ccx, components.as_slice(), false);
        let sym = token::gensym("vtable");
        let vt_gvar = format!("vtable{}", sym.uint()).with_c_str(|buf| {
            llvm::LLVMAddGlobal(ccx.llmod(), val_ty(tbl).to_ref(), buf)
        });
        llvm::LLVMSetInitializer(vt_gvar, tbl);
        llvm::LLVMSetGlobalConstant(vt_gvar, llvm::True);
        llvm::SetLinkage(vt_gvar, llvm::InternalLinkage);
        vt_gvar
    }
}

fn emit_vtable_methods(bcx: Block,
                       impl_id: ast::DefId,
                       substs: subst::Substs)
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
        let ident = ty::impl_or_trait_item(tcx, method_def_id).ident();
        // The substitutions we have are on the impl, so we grab
        // the method type from the impl to substitute into.
        let m_id = method_with_name(ccx, impl_id, ident.name);
        let ti = ty::impl_or_trait_item(tcx, m_id);
        match ti {
            ty::MethodTraitItem(m) => {
                debug!("(making impl vtable) emitting method {} at subst {}",
                       m.repr(tcx),
                       substs.repr(tcx));
                if m.generics.has_type_params(subst::FnSpace) ||
                   ty::type_has_self(ty::mk_bare_fn(tcx, m.fty.clone())) {
                    debug!("(making impl vtable) method has self or type \
                            params: {}",
                           token::get_ident(ident));
                    Some(C_null(Type::nil(ccx).ptr_to())).into_iter()
                } else {
                    let mut fn_ref = trans_fn_ref_with_substs(
                        bcx,
                        m_id,
                        ExprId(0),
                        substs.clone());
                    if m.explicit_self == ty::ByValueExplicitSelfCategory {
                        fn_ref = trans_unboxing_shim(bcx,
                                                     fn_ref,
                                                     &m.fty,
                                                     m_id,
                                                     substs.clone());
                    }
                    Some(fn_ref).into_iter()
                }
            }
            ty::TypeTraitItem(_) => {
                None.into_iter()
            }
        }
    }).collect()
}

pub fn trans_trait_cast<'blk, 'tcx>(bcx: Block<'blk, 'tcx>,
                                    datum: Datum<Expr>,
                                    id: ast::NodeId,
                                    trait_ref: Rc<ty::TraitRef>,
                                    dest: expr::Dest)
                                    -> Block<'blk, 'tcx> {
    /*!
     * Generates the code to convert from a pointer (`Box<T>`, `&T`, etc)
     * into an object (`Box<Trait>`, `&Trait`, etc). This means creating a
     * pair where the first word is the vtable and the second word is
     * the pointer.
     */

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
    let llboxdest = GEPi(bcx, lldest, [0u, abi::trt_field_box]);
    let llboxdest = PointerCast(bcx, llboxdest, llbox_ty.ptr_to());
    bcx = datum.store_to(bcx, llboxdest);

    // Store the vtable into the second half of pair.
    let vtable = get_vtable(bcx, datum_ty, trait_ref);
    let llvtabledest = GEPi(bcx, lldest, [0u, abi::trt_field_vtable]);
    let llvtabledest = PointerCast(bcx, llvtabledest, val_ty(vtable).ptr_to());
    Store(bcx, vtable, llvtabledest);

    bcx
}
