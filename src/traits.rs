use rustc::traits::{self, Reveal, SelectionContext};

use eval_context::EvalContext;
use memory::Pointer;

use rustc::hir::def_id::DefId;
use rustc::ty::fold::TypeFoldable;
use rustc::ty::subst::Substs;
use rustc::ty::{self, Ty, TyCtxt};
use syntax::codemap::DUMMY_SP;
use syntax::ast;

use error::EvalResult;

impl<'a, 'tcx> EvalContext<'a, 'tcx> {

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
    pub fn get_vtable(&mut self, ty: Ty<'tcx>, trait_ref: ty::PolyTraitRef<'tcx>) -> EvalResult<'tcx, Pointer> {
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
                            self.memory.create_fn_ptr(mth.method.def_id, mth.substs)
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
                    let instance = ::eval_context::resolve_closure(self.tcx, closure_def_id, substs, ty::ClosureKind::FnOnce);
                    vec![Some(self.memory.create_fn_alloc(instance))].into_iter()
                }

                // turn a function definition into a Fn trait object
                traits::VtableFnPointer(traits::VtableFnPointerData { fn_ty, .. }) => {
                    match fn_ty.sty {
                        ty::TyFnDef(did, substs, _) => {
                            let instance = ty::Instance {
                                def: ty::InstanceDef::FnPtrShim(did, fn_ty),
                                substs,
                            };
                            vec![Some(self.memory.create_fn_alloc(instance))].into_iter()
                        },
                        ty::TyFnPtr(_) => {
                            unimplemented!();
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
        let drop_in_place = self.tcx.lang_items.drop_in_place_fn().expect("drop_in_place lang item not available");
        if let ty::TyAdt(adt_def, substs) = trait_ref.self_ty().sty {
            if adt_def.has_dtor(self.tcx) {
                let env = self.tcx.empty_parameter_environment();
                let def = if self.tcx.type_needs_drop_given_env(ty, &env) {
                    ty::InstanceDef::DropGlue(drop_in_place, Some(ty))
                } else {
                    ty::InstanceDef::DropGlue(drop_in_place, None)
                };
                let instance = ty::Instance { substs, def };
                let fn_ptr = self.memory.create_fn_alloc(instance);
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

    pub fn read_size_and_align_from_vtable(&self, vtable: Pointer) -> EvalResult<'tcx, (u64, u64)> {
        let pointer_size = self.memory.pointer_size();
        let size = self.memory.read_usize(vtable.offset(pointer_size))?;
        let align = self.memory.read_usize(vtable.offset(pointer_size * 2))?;
        Ok((size, align))
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
    ) -> ty::Instance<'tcx> {
        if let Some(trait_id) = self.tcx.trait_of_item(def_id) {
            let trait_ref = ty::Binder(ty::TraitRef::new(trait_id, substs));
            let vtable = self.fulfill_obligation(trait_ref);
            if let traits::VtableImpl(vtable_impl) = vtable {
                let name = self.tcx.item_name(def_id);
                let assoc_const_opt = self.tcx.associated_items(vtable_impl.impl_def_id)
                    .find(|item| item.kind == ty::AssociatedKind::Const && item.name == name);
                if let Some(assoc_const) = assoc_const_opt {
                    return ty::Instance::new(assoc_const.def_id, vtable_impl.substs);
                }
            }
        }
        ty::Instance::new(def_id, substs)
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
