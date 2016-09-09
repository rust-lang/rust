use rustc::hir::def_id::DefId;
use rustc::traits::{self, Reveal, SelectionContext};
use rustc::ty::subst::{Substs, Subst};
use rustc::ty;

use super::EvalContext;
use error::EvalResult;
use memory::Pointer;
use super::terminator::{get_impl_method, ImplMethod};

impl<'a, 'tcx> EvalContext<'a, 'tcx> {
    /// Creates a returns a dynamic vtable for the given type and vtable origin.
    /// This is used only for objects.
    ///
    /// The `trait_ref` encodes the erased self type. Hence if we are
    /// making an object `Foo<Trait>` from a value of type `Foo<T>`, then
    /// `trait_ref` would map `T:Trait`.
    pub fn get_vtable(&mut self, trait_ref: ty::PolyTraitRef<'tcx>) -> EvalResult<'tcx, Pointer> {
        let tcx = self.tcx;

        debug!("get_vtable(trait_ref={:?})", trait_ref);

        let methods: Vec<_> = traits::supertraits(tcx, trait_ref.clone()).flat_map(|trait_ref| {
            match self.fulfill_obligation(trait_ref.clone()) {
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
                    self.get_vtable_methods(id, substs)
                        .into_iter()
                        .map(|opt_mth| opt_mth.map(|mth| {
                            self.memory.create_fn_ptr(mth.method.def_id, mth.substs, mth.method.fty)
                        }))
                        .collect::<Vec<_>>()
                        .into_iter()
                }
                traits::VtableClosure(
                    traits::VtableClosureData {
                        closure_def_id,
                        substs,
                        nested: _ }) => {
                    let closure_type = self.tcx.closure_type(closure_def_id, substs);
                    vec![Some(self.memory.create_closure_ptr(closure_def_id, substs, closure_type))].into_iter()
                }
                traits::VtableFnPointer(
                    traits::VtableFnPointerData {
                        fn_ty: bare_fn_ty,
                        nested: _ }) => {
                    let trait_closure_kind = tcx.lang_items.fn_trait_kind(trait_ref.def_id()).unwrap();
                    //vec![trans_fn_pointer_shim(ccx, trait_closure_kind, bare_fn_ty)].into_iter()
                    unimplemented!()
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

        let size = self.type_size(trait_ref.self_ty());
        let align = self.type_align(trait_ref.self_ty());

        let ptr_size = self.memory.pointer_size();
        let vtable = self.memory.allocate(ptr_size * (3 + methods.len()), ptr_size)?;

        // FIXME: generate a destructor for the vtable.
        // trans does this with glue::get_drop_glue(ccx, trait_ref.self_ty())

        self.memory.write_usize(vtable.offset(ptr_size as isize), size as u64)?;
        self.memory.write_usize(vtable.offset((ptr_size * 2) as isize), align as u64)?;

        for (i, method) in methods.into_iter().enumerate() {
            if let Some(method) = method {
                self.memory.write_ptr(vtable.offset(ptr_size as isize * (3 + i as isize)), method)?;
            }
        }

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

        let trait_item_def_ids = self.tcx.trait_item_def_ids(trait_id);
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

                let trait_method_type = match self.tcx.impl_or_trait_item(trait_method_def_id) {
                    ty::MethodTraitItem(m) => m,
                    _ => bug!("should be a method, not other assoc item"),
                };
                let name = trait_method_type.name;

                // Some methods cannot be called on an object; skip those.
                if !self.tcx.is_vtable_safe_method(trait_id, &trait_method_type) {
                    debug!("get_vtable_methods: not vtable safe");
                    return None;
                }

                debug!("get_vtable_methods: trait_method_type={:?}",
                       trait_method_type);

                // the method may have some early-bound lifetimes, add
                // regions for those
                let method_substs = Substs::for_item(self.tcx, trait_method_def_id,
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
                    let predicates = mth.method.predicates.predicates.subst(self.tcx, &mth.substs);
                    if !self.normalize_and_test_predicates(predicates) {
                        debug!("get_vtable_methods: predicates do not hold");
                        return None;
                    }
                }

                Some(mth)
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

        self.tcx.infer_ctxt(None, None, Reveal::All).enter(|infcx| {
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
