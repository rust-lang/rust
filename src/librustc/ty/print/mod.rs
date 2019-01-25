use crate::hir::map::DefPathData;
use crate::hir::def_id::{CrateNum, DefId};
use crate::ty::{self, DefIdTree, Ty, TyCtxt};
use crate::ty::subst::{Kind, Subst, SubstsRef};

use rustc_data_structures::fx::FxHashSet;

// `pretty` is a separate module only for organization.
mod pretty;
pub use self::pretty::*;

pub trait Print<'gcx, 'tcx, P> {
    type Output;
    type Error;

    fn print(&self, cx: P) -> Result<Self::Output, Self::Error>;
}

pub trait Printer<'gcx: 'tcx, 'tcx>: Sized {
    type Error;

    type Path;
    type Region;
    type Type;
    type DynExistential;

    fn tcx(&'a self) -> TyCtxt<'a, 'gcx, 'tcx>;

    fn print_def_path(
        self,
        def_id: DefId,
        substs: Option<SubstsRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        self.default_print_def_path(def_id, substs)
    }
    fn print_impl_path(
        self,
        impl_def_id: DefId,
        substs: Option<SubstsRef<'tcx>>,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        self.default_print_impl_path(impl_def_id, substs, self_ty, trait_ref)
    }

    fn print_region(
        self,
        region: ty::Region<'_>,
    ) -> Result<Self::Region, Self::Error>;

    fn print_type(
        self,
        ty: Ty<'tcx>,
    ) -> Result<Self::Type, Self::Error>;

    fn print_dyn_existential(
        self,
        predicates: &'tcx ty::List<ty::ExistentialPredicate<'tcx>>,
    ) -> Result<Self::DynExistential, Self::Error>;

    fn path_crate(
        self,
        cnum: CrateNum,
    ) -> Result<Self::Path, Self::Error>;
    fn path_qualified(
        self,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error>;

    fn path_append_impl(
        self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error>;
    fn path_append(
        self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        text: &str,
    ) -> Result<Self::Path, Self::Error>;
    fn path_generic_args(
        self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        args: &[Kind<'tcx>],
    ) -> Result<Self::Path, Self::Error>;

    // Defaults (should not be overriden):

    fn default_print_def_path(
        self,
        def_id: DefId,
        substs: Option<SubstsRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        debug!("default_print_def_path: def_id={:?}, substs={:?}", def_id, substs);
        let key = self.tcx().def_key(def_id);
        debug!("default_print_def_path: key={:?}", key);

        match key.disambiguated_data.data {
            DefPathData::CrateRoot => {
                assert!(key.parent.is_none());
                self.path_crate(def_id.krate)
            }

            DefPathData::Impl => {
                let mut self_ty = self.tcx().type_of(def_id);
                if let Some(substs) = substs {
                    self_ty = self_ty.subst(self.tcx(), substs);
                }

                let mut impl_trait_ref = self.tcx().impl_trait_ref(def_id);
                if let Some(substs) = substs {
                    impl_trait_ref = impl_trait_ref.subst(self.tcx(), substs);
                }
                self.print_impl_path(def_id, substs, self_ty, impl_trait_ref)
            }

            _ => {
                let generics = substs.map(|_| self.tcx().generics_of(def_id));
                let generics_parent = generics.as_ref().and_then(|g| g.parent);
                let parent_def_id = DefId { index: key.parent.unwrap(), ..def_id };
                let print_parent_path = |cx: Self| {
                    if let Some(generics_parent_def_id) = generics_parent {
                        assert_eq!(parent_def_id, generics_parent_def_id);

                        // FIXME(eddyb) try to move this into the parent's printing
                        // logic, instead of doing it when printing the child.
                        let parent_generics = cx.tcx().generics_of(parent_def_id);
                        let parent_has_own_self =
                            parent_generics.has_self && parent_generics.parent_count == 0;
                        if let (Some(substs), true) = (substs, parent_has_own_self) {
                            let trait_ref = ty::TraitRef::new(parent_def_id, substs);
                            cx.path_qualified(trait_ref.self_ty(), Some(trait_ref))
                        } else {
                            cx.print_def_path(parent_def_id, substs)
                        }
                    } else {
                        cx.print_def_path(parent_def_id, None)
                    }
                };
                let print_path = |cx: Self| {
                    match key.disambiguated_data.data {
                        // Skip `::{{constructor}}` on tuple/unit structs.
                        DefPathData::StructCtor => print_parent_path(cx),

                        _ => {
                            cx.path_append(
                                print_parent_path,
                                &key.disambiguated_data.data.as_interned_str().as_str(),
                            )
                        }
                    }
                };

                if let (Some(generics), Some(substs)) = (generics, substs) {
                    let args = self.generic_args_to_print(generics, substs);
                    self.path_generic_args(print_path, args)
                } else {
                    print_path(self)
                }
            }
        }
    }

    fn generic_args_to_print(
        &self,
        generics: &'tcx ty::Generics,
        substs: SubstsRef<'tcx>,
    ) -> &'tcx [Kind<'tcx>] {
        let mut own_params = generics.parent_count..generics.count();

        // Don't print args for `Self` parameters (of traits).
        if generics.has_self && own_params.start == 0 {
            own_params.start = 1;
        }

        // Don't print args that are the defaults of their respective parameters.
        own_params.end -= generics.params.iter().rev().take_while(|param| {
            match param.kind {
                ty::GenericParamDefKind::Lifetime => false,
                ty::GenericParamDefKind::Type { has_default, .. } => {
                    has_default && substs[param.index as usize] == Kind::from(
                        self.tcx().type_of(param.def_id).subst(self.tcx(), substs)
                    )
                }
                ty::GenericParamDefKind::Const => false, // FIXME(const_generics:defaults)
            }
        }).count();

        &substs[own_params]
    }

    fn default_print_impl_path(
        self,
        impl_def_id: DefId,
        _substs: Option<SubstsRef<'tcx>>,
        self_ty: Ty<'tcx>,
        impl_trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        debug!("default_print_impl_path: impl_def_id={:?}, self_ty={}, impl_trait_ref={:?}",
               impl_def_id, self_ty, impl_trait_ref);

        // Decide whether to print the parent path for the impl.
        // Logically, since impls are global, it's never needed, but
        // users may find it useful. Currently, we omit the parent if
        // the impl is either in the same module as the self-type or
        // as the trait.
        let parent_def_id = self.tcx().parent(impl_def_id).unwrap();
        let in_self_mod = match characteristic_def_id_of_type(self_ty) {
            None => false,
            Some(ty_def_id) => self.tcx().parent(ty_def_id) == Some(parent_def_id),
        };
        let in_trait_mod = match impl_trait_ref {
            None => false,
            Some(trait_ref) => self.tcx().parent(trait_ref.def_id) == Some(parent_def_id),
        };

        if !in_self_mod && !in_trait_mod {
            // If the impl is not co-located with either self-type or
            // trait-type, then fallback to a format that identifies
            // the module more clearly.
            self.path_append_impl(
                |cx| cx.print_def_path(parent_def_id, None),
                self_ty,
                impl_trait_ref,
            )
        } else {
            // Otherwise, try to give a good form that would be valid language
            // syntax. Preferably using associated item notation.
            self.path_qualified(self_ty, impl_trait_ref)
        }
    }
}

/// As a heuristic, when we see an impl, if we see that the
/// 'self type' is a type defined in the same module as the impl,
/// we can omit including the path to the impl itself. This
/// function tries to find a "characteristic `DefId`" for a
/// type. It's just a heuristic so it makes some questionable
/// decisions and we may want to adjust it later.
pub fn characteristic_def_id_of_type(ty: Ty<'_>) -> Option<DefId> {
    match ty.sty {
        ty::Adt(adt_def, _) => Some(adt_def.did),

        ty::Dynamic(data, ..) => data.principal_def_id(),

        ty::Array(subty, _) |
        ty::Slice(subty) => characteristic_def_id_of_type(subty),

        ty::RawPtr(mt) => characteristic_def_id_of_type(mt.ty),

        ty::Ref(_, ty, _) => characteristic_def_id_of_type(ty),

        ty::Tuple(ref tys) => tys.iter()
                                   .filter_map(|ty| characteristic_def_id_of_type(ty))
                                   .next(),

        ty::FnDef(def_id, _) |
        ty::Closure(def_id, _) |
        ty::Generator(def_id, _, _) |
        ty::Foreign(def_id) => Some(def_id),

        ty::Bool |
        ty::Char |
        ty::Int(_) |
        ty::Uint(_) |
        ty::Str |
        ty::FnPtr(_) |
        ty::Projection(_) |
        ty::Placeholder(..) |
        ty::UnnormalizedProjection(..) |
        ty::Param(_) |
        ty::Opaque(..) |
        ty::Infer(_) |
        ty::Bound(..) |
        ty::Error |
        ty::GeneratorWitness(..) |
        ty::Never |
        ty::Float(_) => None,
    }
}

impl<'gcx: 'tcx, 'tcx, P: Printer<'gcx, 'tcx>> Print<'gcx, 'tcx, P> for ty::RegionKind {
    type Output = P::Region;
    type Error = P::Error;
    fn print(&self, cx: P) -> Result<Self::Output, Self::Error> {
        cx.print_region(self)
    }
}

impl<'gcx: 'tcx, 'tcx, P: Printer<'gcx, 'tcx>> Print<'gcx, 'tcx, P> for ty::Region<'_> {
    type Output = P::Region;
    type Error = P::Error;
    fn print(&self, cx: P) -> Result<Self::Output, Self::Error> {
        cx.print_region(self)
    }
}

impl<'gcx: 'tcx, 'tcx, P: Printer<'gcx, 'tcx>> Print<'gcx, 'tcx, P> for Ty<'tcx> {
    type Output = P::Type;
    type Error = P::Error;
    fn print(&self, cx: P) -> Result<Self::Output, Self::Error> {
        cx.print_type(self)
    }
}

impl<'gcx: 'tcx, 'tcx, P: Printer<'gcx, 'tcx>> Print<'gcx, 'tcx, P>
    for &'tcx ty::List<ty::ExistentialPredicate<'tcx>>
{
    type Output = P::DynExistential;
    type Error = P::Error;
    fn print(&self, cx: P) -> Result<Self::Output, Self::Error> {
        cx.print_dyn_existential(self)
    }
}
