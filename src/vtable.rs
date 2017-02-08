use rustc::hir::def_id::DefId;
use rustc::traits::{self, Reveal, SelectionContext};
use rustc::ty::subst::Substs;
use rustc::ty;

use error::EvalResult;
use eval_context::EvalContext;
use memory::Pointer;
use terminator::{get_impl_method, ImplMethod};

impl<'a, 'tcx> EvalContext<'a, 'tcx> {
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
                let fn_ty = match  self.tcx.item_type(drop_def_id).sty {
                    ty::TyFnDef(_, _, fn_ty) => self.tcx.erase_regions(&fn_ty),
                    _ => bug!("drop method is not a TyFnDef"),
                };
                let fn_ptr = self.memory.create_drop_glue(self.tcx, drop_def_id, substs, fn_ty);
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

        self.memory.mark_static(vtable.alloc_id, false)?;

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
}
