use rustc::traits::{self, Reveal, SelectionContext};

use eval_context::EvalContext;
use memory::Pointer;

use rustc::hir::def_id::DefId;
use rustc::ty::fold::TypeFoldable;
use rustc::ty::subst::Substs;
use rustc::ty::{self, Ty, TyCtxt};
use syntax::codemap::DUMMY_SP;
use syntax::{ast, abi};

use error::{EvalError, EvalResult};
use memory::Function;
use value::PrimVal;
use value::Value;

impl<'a, 'tcx> EvalContext<'a, 'tcx> {
    /// Trait method, which has to be resolved to an impl method.
    pub(crate) fn trait_method(
        &mut self,
        trait_id: DefId,
        def_id: DefId,
        substs: &'tcx Substs<'tcx>,
        args: &mut Vec<(Value, Ty<'tcx>)>,
    ) -> EvalResult<'tcx, (DefId, &'tcx Substs<'tcx>, Vec<(Pointer, Ty<'tcx>)>)> {
        let trait_ref = ty::TraitRef::from_method(self.tcx, trait_id, substs);
        let trait_ref = self.tcx.normalize_associated_type(&ty::Binder(trait_ref));

        match self.fulfill_obligation(trait_ref) {
            traits::VtableImpl(vtable_impl) => {
                let impl_did = vtable_impl.impl_def_id;
                let mname = self.tcx.item_name(def_id);
                // Create a concatenated set of substitutions which includes those from the impl
                // and those from the method:
                let (did, substs) = find_method(self.tcx, substs, impl_did, vtable_impl.substs, mname);

                Ok((did, substs, Vec::new()))
            }

            traits::VtableClosure(vtable_closure) => {
                let trait_closure_kind = self.tcx
                    .lang_items
                    .fn_trait_kind(trait_id)
                    .expect("The substitutions should have no type parameters remaining after passing through fulfill_obligation");
                let closure_kind = self.tcx.closure_kind(vtable_closure.closure_def_id);
                trace!("closures {:?}, {:?}", closure_kind, trait_closure_kind);
                self.unpack_fn_args(args)?;
                let mut temporaries = Vec::new();
                match (closure_kind, trait_closure_kind) {
                    (ty::ClosureKind::Fn, ty::ClosureKind::Fn) |
                    (ty::ClosureKind::FnMut, ty::ClosureKind::FnMut) |
                    (ty::ClosureKind::FnOnce, ty::ClosureKind::FnOnce) |
                    (ty::ClosureKind::Fn, ty::ClosureKind::FnMut) => {} // No adapter needed.

                    (ty::ClosureKind::Fn, ty::ClosureKind::FnOnce) |
                    (ty::ClosureKind::FnMut, ty::ClosureKind::FnOnce) => {
                        // The closure fn is a `fn(&self, ...)` or `fn(&mut self, ...)`.
                        // We want a `fn(self, ...)`.
                        // We can produce this by doing something like:
                        //
                        //     fn call_once(self, ...) { call_mut(&self, ...) }
                        //     fn call_once(mut self, ...) { call_mut(&mut self, ...) }
                        //
                        // These are both the same at trans time.

                        // Interpreter magic: insert an intermediate pointer, so we can skip the
                        // intermediate function call.
                        let ptr = match args[0].0 {
                            Value::ByRef(ptr) => ptr,
                            Value::ByVal(primval) => {
                                let ptr = self.alloc_ptr(args[0].1)?;
                                let size = self.type_size(args[0].1)?.expect("closures are sized");
                                self.memory.write_primval(ptr, primval, size)?;
                                ptr
                            },
                            Value::ByValPair(a, b) => {
                                let ptr = self.alloc_ptr(args[0].1)?;
                                self.write_pair_to_ptr(a, b, ptr, args[0].1)?;
                                ptr
                            },
                        };
                        temporaries.push((ptr, args[0].1));
                        args[0].0 = Value::ByVal(PrimVal::Ptr(ptr));
                        args[0].1 = self.tcx.mk_mut_ptr(args[0].1);
                    }

                    _ => bug!("cannot convert {:?} to {:?}", closure_kind, trait_closure_kind),
                }
                Ok((vtable_closure.closure_def_id, vtable_closure.substs.substs, temporaries))
            }

            traits::VtableFnPointer(vtable_fn_ptr) => {
                if let ty::TyFnDef(did, substs, _) = vtable_fn_ptr.fn_ty.sty {
                    args.remove(0);
                    self.unpack_fn_args(args)?;
                    Ok((did, substs, Vec::new()))
                } else {
                    bug!("VtableFnPointer did not contain a concrete function: {:?}", vtable_fn_ptr)
                }
            }

            traits::VtableObject(ref data) => {
                let idx = self.tcx.get_vtable_index_of_object_method(data, def_id) as u64;
                if args.is_empty() {
                    return Err(EvalError::VtableForArgumentlessMethod);
                }
                let (self_ptr, vtable) = args[0].0.expect_ptr_vtable_pair(&self.memory)?;
                let idx = idx + 3;
                let offset = idx * self.memory.pointer_size();
                let fn_ptr = self.memory.read_ptr(vtable.offset(offset))?;
                trace!("args: {:#?}", args);
                match self.memory.get_fn(fn_ptr.alloc_id)? {
                    Function::FnDefAsTraitObject(fn_def) => {
                        trace!("sig: {:#?}", fn_def.sig);
                        assert!(fn_def.abi != abi::Abi::RustCall);
                        assert_eq!(args.len(), 2);
                        // a function item turned into a closure trait object
                        // the first arg is just there to give use the vtable
                        args.remove(0);
                        self.unpack_fn_args(args)?;
                        Ok((fn_def.def_id, fn_def.substs, Vec::new()))
                    },
                    Function::DropGlue(_) => Err(EvalError::ManuallyCalledDropGlue),
                    Function::Concrete(fn_def) => {
                        trace!("sig: {:#?}", fn_def.sig);
                        args[0] = (
                            Value::ByVal(PrimVal::Ptr(self_ptr)),
                            fn_def.sig.inputs()[0],
                        );
                        Ok((fn_def.def_id, fn_def.substs, Vec::new()))
                    },
                    Function::Closure(fn_def) => {
                        self.unpack_fn_args(args)?;
                        Ok((fn_def.def_id, fn_def.substs, Vec::new()))
                    }
                    Function::FnPtrAsTraitObject(sig) => {
                        trace!("sig: {:#?}", sig);
                        // the first argument was the fat ptr
                        args.remove(0);
                        self.unpack_fn_args(args)?;
                        let fn_ptr = self.memory.read_ptr(self_ptr)?;
                        let fn_def = self.memory.get_fn(fn_ptr.alloc_id)?.expect_concrete()?;
                        assert_eq!(sig, fn_def.sig);
                        Ok((fn_def.def_id, fn_def.substs, Vec::new()))
                    }
                }
            },
            vtable => bug!("resolved vtable bad vtable {:?} in trans", vtable),
        }
    }

    pub(crate) fn fulfill_obligation(&self, trait_ref: ty::PolyTraitRef<'tcx>) -> traits::Vtable<'tcx, ()> {
        // Do the initial selection for the obligation. This yields the shallow result we are
        // looking for -- that is, what specific impl.
        self.tcx.infer_ctxt((), Reveal::All).enter(|infcx| {
            let mut selcx = traits::SelectionContext::new(&infcx);

            let obligation = traits::Obligation::new(
                traits::ObligationCause::misc(DUMMY_SP, ast::DUMMY_NODE_ID),
                trait_ref.to_poly_trait_predicate(),
            );
            let selection = selcx.select(&obligation).unwrap().unwrap();

            // Currently, we use a fulfillment context to completely resolve all nested obligations.
            // This is because they can inform the inference of the impl's type parameters.
            let mut fulfill_cx = traits::FulfillmentContext::new();
            let vtable = selection.map(|predicate| {
                fulfill_cx.register_predicate_obligation(&infcx, predicate);
            });
            infcx.drain_fulfillment_cx_or_panic(DUMMY_SP, &mut fulfill_cx, &vtable)
        })
    }

    /// Creates a dynamic vtable for the given type and vtable origin. This is used only for
    /// objects.
    ///
    /// The `trait_ref` encodes the erased self type. Hence if we are
    /// making an object `Foo<Trait>` from a value of type `Foo<T>`, then
    /// `trait_ref` would map `T:Trait`.
    pub fn get_vtable(&mut self, trait_ref: ty::PolyTraitRef<'tcx>) -> EvalResult<'tcx, Pointer> {
        let tcx = self.tcx;

        debug!("get_vtable(trait_ref={:?})", trait_ref);

        let methods: Vec<_> = traits::supertraits(tcx, trait_ref).flat_map(|trait_ref| {
            match self.fulfill_obligation(trait_ref) {
                // Should default trait error here?
                traits::VtableDefaultImpl(_) |
                traits::VtableBuiltin(_) => {
                    Vec::new().into_iter()
                }

                traits::VtableImpl(traits::VtableImplData { impl_def_id: id, substs, .. }) => {
                    self.get_vtable_methods(id, substs)
                        .into_iter()
                        .map(|opt_mth| opt_mth.map(|mth| {
                            let fn_ty = self.tcx.item_type(mth.method.def_id);
                            let fn_ty = match fn_ty.sty {
                                ty::TyFnDef(_, _, fn_ty) => fn_ty,
                                _ => bug!("bad function type: {}", fn_ty),
                            };
                            let fn_ty = self.tcx.erase_regions(&fn_ty);
                            self.memory.create_fn_ptr(self.tcx, mth.method.def_id, mth.substs, fn_ty)
                        }))
                        .collect::<Vec<_>>()
                        .into_iter()
                }

                traits::VtableClosure(
                    traits::VtableClosureData {
                        closure_def_id,
                        substs,
                        ..
                    }
                ) => {
                    let closure_type = self.tcx.closure_type(closure_def_id, substs);
                    vec![Some(self.memory.create_closure_ptr(self.tcx, closure_def_id, substs, closure_type))].into_iter()
                }

                // turn a function definition into a Fn trait object
                traits::VtableFnPointer(traits::VtableFnPointerData { fn_ty, .. }) => {
                    match fn_ty.sty {
                        ty::TyFnDef(did, substs, bare_fn_ty) => {
                            vec![Some(self.memory.create_fn_as_trait_glue(self.tcx, did, substs, bare_fn_ty))].into_iter()
                        },
                        ty::TyFnPtr(bare_fn_ty) => {
                            vec![Some(self.memory.create_fn_ptr_as_trait_glue(bare_fn_ty))].into_iter()
                        },
                        _ => bug!("bad VtableFnPointer fn_ty: {:#?}", fn_ty.sty),
                    }
                }

                traits::VtableObject(ref data) => {
                    // this would imply that the Self type being erased is
                    // an object type; this cannot happen because we
                    // cannot cast an unsized type into a trait object
                    bug!("cannot get vtable for an object type: {:?}",
                         data);
                }

                vtable @ traits::VtableParam(..) => {
                    bug!("resolved vtable for {:?} to bad vtable {:?} in trans",
                         trait_ref,
                         vtable);
                }
            }
        }).collect();

        let size = self.type_size(trait_ref.self_ty())?.expect("can't create a vtable for an unsized type");
        let align = self.type_align(trait_ref.self_ty())?;

        let ptr_size = self.memory.pointer_size();
        let vtable = self.memory.allocate(ptr_size * (3 + methods.len() as u64), ptr_size)?;

        // in case there is no drop function to be called, this still needs to be initialized
        self.memory.write_usize(vtable, 0)?;
        if let ty::TyAdt(adt_def, substs) = trait_ref.self_ty().sty {
            if let Some(drop_def_id) = adt_def.destructor() {
                let fn_ty = match self.tcx.item_type(drop_def_id).sty {
                    ty::TyFnDef(_, _, fn_ty) => self.tcx.erase_regions(&fn_ty),
                    _ => bug!("drop method is not a TyFnDef"),
                };
                // The real type is taken from the self argument in `fn drop(&mut self)`
                let real_ty = match fn_ty.sig.skip_binder().inputs()[0].sty {
                    ty::TyRef(_, mt) => self.monomorphize(mt.ty, substs),
                    _ => bug!("first argument of Drop::drop must be &mut T"),
                };
                let fn_ptr = self.memory.create_drop_glue(real_ty);
                self.memory.write_ptr(vtable, fn_ptr)?;
            }
        }

        self.memory.write_usize(vtable.offset(ptr_size), size)?;
        self.memory.write_usize(vtable.offset(ptr_size * 2), align)?;

        for (i, method) in methods.into_iter().enumerate() {
            if let Some(method) = method {
                self.memory.write_ptr(vtable.offset(ptr_size * (3 + i as u64)), method)?;
            }
        }

        self.memory.mark_static_initalized(vtable.alloc_id, false)?;

        Ok(vtable)
    }

    fn get_vtable_methods(&mut self, impl_id: DefId, substs: &'tcx Substs<'tcx>) -> Vec<Option<ImplMethod<'tcx>>> {
        debug!("get_vtable_methods(impl_id={:?}, substs={:?}", impl_id, substs);

        let trait_id = match self.tcx.impl_trait_ref(impl_id) {
            Some(t_id) => t_id.def_id,
            None       => bug!("make_impl_vtable: don't know how to \
                                make a vtable for a type impl!")
        };

        self.tcx.populate_implementations_for_trait_if_necessary(trait_id);

        self.tcx
            .associated_items(trait_id)
            // Filter out non-method items.
            .filter_map(|trait_method_type| {
                if trait_method_type.kind != ty::AssociatedKind::Method {
                    return None;
                }
                debug!("get_vtable_methods: trait_method_type={:?}",
                       trait_method_type);

                let name = trait_method_type.name;

                // Some methods cannot be called on an object; skip those.
                if !self.tcx.is_vtable_safe_method(trait_id, &trait_method_type) {
                    debug!("get_vtable_methods: not vtable safe");
                    return Some(None);
                }

                debug!("get_vtable_methods: trait_method_type={:?}",
                       trait_method_type);

                // the method may have some early-bound lifetimes, add
                // regions for those
                let method_substs = Substs::for_item(self.tcx, trait_method_type.def_id,
                                                     |_, _| self.tcx.mk_region(ty::ReErased),
                                                     |_, _| self.tcx.types.err);

                // The substitutions we have are on the impl, so we grab
                // the method type from the impl to substitute into.
                let mth = get_impl_method(self.tcx, method_substs, impl_id, substs, name);

                debug!("get_vtable_methods: mth={:?}", mth);

                // If this is a default method, it's possible that it
                // relies on where clauses that do not hold for this
                // particular set of type parameters. Note that this
                // method could then never be called, so we do not want to
                // try and trans it, in that case. Issue #23435.
                if mth.is_provided {
                    let predicates = self.tcx.item_predicates(trait_method_type.def_id).instantiate_own(self.tcx, mth.substs);
                    if !self.normalize_and_test_predicates(predicates.predicates) {
                        debug!("get_vtable_methods: predicates do not hold");
                        return Some(None);
                    }
                }

                Some(Some(mth))
            })
            .collect()
    }

    /// Normalizes the predicates and checks whether they hold.  If this
    /// returns false, then either normalize encountered an error or one
    /// of the predicates did not hold. Used when creating vtables to
    /// check for unsatisfiable methods.
    fn normalize_and_test_predicates(&mut self, predicates: Vec<ty::Predicate<'tcx>>) -> bool {
        debug!("normalize_and_test_predicates(predicates={:?})",
               predicates);

        self.tcx.infer_ctxt((), Reveal::All).enter(|infcx| {
            let mut selcx = SelectionContext::new(&infcx);
            let mut fulfill_cx = traits::FulfillmentContext::new();
            let cause = traits::ObligationCause::dummy();
            let traits::Normalized { value: predicates, obligations } =
                traits::normalize(&mut selcx, cause.clone(), &predicates);
            for obligation in obligations {
                fulfill_cx.register_predicate_obligation(&infcx, obligation);
            }
            for predicate in predicates {
                let obligation = traits::Obligation::new(cause.clone(), predicate);
                fulfill_cx.register_predicate_obligation(&infcx, obligation);
            }

            fulfill_cx.select_all_or_error(&infcx).is_ok()
        })
    }

    pub(crate) fn resolve_associated_const(
        &self,
        def_id: DefId,
        substs: &'tcx Substs<'tcx>,
    ) -> (DefId, &'tcx Substs<'tcx>) {
        if let Some(trait_id) = self.tcx.trait_of_item(def_id) {
            let trait_ref = ty::Binder(ty::TraitRef::new(trait_id, substs));
            let vtable = self.fulfill_obligation(trait_ref);
            if let traits::VtableImpl(vtable_impl) = vtable {
                let name = self.tcx.item_name(def_id);
                let assoc_const_opt = self.tcx.associated_items(vtable_impl.impl_def_id)
                    .find(|item| item.kind == ty::AssociatedKind::Const && item.name == name);
                if let Some(assoc_const) = assoc_const_opt {
                    return (assoc_const.def_id, vtable_impl.substs);
                }
            }
        }
        (def_id, substs)
    }
}

#[derive(Debug)]
pub(super) struct ImplMethod<'tcx> {
    pub(super) method: ty::AssociatedItem,
    pub(super) substs: &'tcx Substs<'tcx>,
    pub(super) is_provided: bool,
}

/// Locates the applicable definition of a method, given its name.
pub(super) fn get_impl_method<'a, 'tcx>(
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    substs: &'tcx Substs<'tcx>,
    impl_def_id: DefId,
    impl_substs: &'tcx Substs<'tcx>,
    name: ast::Name,
) -> ImplMethod<'tcx> {
    assert!(!substs.needs_infer());

    let trait_def_id = tcx.trait_id_of_impl(impl_def_id).unwrap();
    let trait_def = tcx.lookup_trait_def(trait_def_id);

    match trait_def.ancestors(impl_def_id).defs(tcx, name, ty::AssociatedKind::Method).next() {
        Some(node_item) => {
            let substs = tcx.infer_ctxt((), Reveal::All).enter(|infcx| {
                let substs = substs.rebase_onto(tcx, trait_def_id, impl_substs);
                let substs = traits::translate_substs(&infcx, impl_def_id,
                                                      substs, node_item.node);
                tcx.lift(&substs).unwrap_or_else(|| {
                    bug!("trans::meth::get_impl_method: translate_substs \
                          returned {:?} which contains inference types/regions",
                         substs);
                })
            });
            ImplMethod {
                method: node_item.item,
                substs,
                is_provided: node_item.node.is_from_trait(),
            }
        }
        None => {
            bug!("method {:?} not found in {:?}", name, impl_def_id)
        }
    }
}

/// Locates the applicable definition of a method, given its name.
pub fn find_method<'a, 'tcx>(tcx: TyCtxt<'a, 'tcx, 'tcx>,
                             substs: &'tcx Substs<'tcx>,
                             impl_def_id: DefId,
                             impl_substs: &'tcx Substs<'tcx>,
                             name: ast::Name)
                             -> (DefId, &'tcx Substs<'tcx>)
{
    assert!(!substs.needs_infer());

    let trait_def_id = tcx.trait_id_of_impl(impl_def_id).unwrap();
    let trait_def = tcx.lookup_trait_def(trait_def_id);

    match trait_def.ancestors(impl_def_id).defs(tcx, name, ty::AssociatedKind::Method).next() {
        Some(node_item) => {
            let substs = tcx.infer_ctxt((), Reveal::All).enter(|infcx| {
                let substs = substs.rebase_onto(tcx, trait_def_id, impl_substs);
                let substs = traits::translate_substs(&infcx, impl_def_id, substs, node_item.node);
                tcx.lift(&substs).unwrap_or_else(|| {
                    bug!("find_method: translate_substs \
                          returned {:?} which contains inference types/regions",
                         substs);
                })
            });
            (node_item.item.def_id, substs)
        }
        None => {
            bug!("method {:?} not found in {:?}", name, impl_def_id)
        }
    }
}
