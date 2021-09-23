use crate::ty::subst::{GenericArg, Subst};
use crate::ty::{self, DefIdTree, Ty, TyCtxt};

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sso::SsoHashSet;
use rustc_hir::def_id::{CrateNum, DefId};
use rustc_hir::definitions::{DefPathData, DisambiguatedDefPathData};

// `pretty` is a separate module only for organization.
mod pretty;
pub use self::pretty::*;

// FIXME(eddyb) false positive, the lifetime parameters are used with `P:  Printer<...>`.
#[allow(unused_lifetimes)]
pub trait Print<'tcx, P> {
    type Output;
    type Error;

    fn print(&self, cx: P) -> Result<Self::Output, Self::Error>;
}

/// Interface for outputting user-facing "type-system entities"
/// (paths, types, lifetimes, constants, etc.) as a side-effect
/// (e.g. formatting, like `PrettyPrinter` implementors do) or by
/// constructing some alternative representation (e.g. an AST),
/// which the associated types allow passing through the methods.
///
/// For pretty-printing/formatting in particular, see `PrettyPrinter`.
//
// FIXME(eddyb) find a better name; this is more general than "printing".
pub trait Printer<'tcx>: Sized {
    type Error;

    type Path;
    type Region;
    type Type;
    type DynExistential;
    type Const;

    fn tcx(&'a self) -> TyCtxt<'tcx>;

    fn print_def_path(
        self,
        def_id: DefId,
        substs: &'tcx [GenericArg<'tcx>],
    ) -> Result<Self::Path, Self::Error> {
        self.default_print_def_path(def_id, substs)
    }

    fn print_impl_path(
        self,
        impl_def_id: DefId,
        substs: &'tcx [GenericArg<'tcx>],
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        self.default_print_impl_path(impl_def_id, substs, self_ty, trait_ref)
    }

    fn print_region(self, region: ty::Region<'_>) -> Result<Self::Region, Self::Error>;

    fn print_type(self, ty: Ty<'tcx>) -> Result<Self::Type, Self::Error>;

    fn print_dyn_existential(
        self,
        predicates: &'tcx ty::List<ty::Binder<'tcx, ty::ExistentialPredicate<'tcx>>>,
    ) -> Result<Self::DynExistential, Self::Error>;

    fn print_const(self, ct: &'tcx ty::Const<'tcx>) -> Result<Self::Const, Self::Error>;

    fn path_crate(self, cnum: CrateNum) -> Result<Self::Path, Self::Error>;

    fn path_qualified(
        self,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error>;

    fn path_append_impl(
        self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        disambiguated_data: &DisambiguatedDefPathData,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error>;

    fn path_append(
        self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        disambiguated_data: &DisambiguatedDefPathData,
    ) -> Result<Self::Path, Self::Error>;

    fn path_generic_args(
        self,
        print_prefix: impl FnOnce(Self) -> Result<Self::Path, Self::Error>,
        args: &[GenericArg<'tcx>],
    ) -> Result<Self::Path, Self::Error>;

    // Defaults (should not be overridden):

    #[instrument(skip(self), level = "debug")]
    fn default_print_def_path(
        self,
        def_id: DefId,
        substs: &'tcx [GenericArg<'tcx>],
    ) -> Result<Self::Path, Self::Error> {
        let key = self.tcx().def_key(def_id);
        debug!(?key);

        match key.disambiguated_data.data {
            DefPathData::CrateRoot => {
                assert!(key.parent.is_none());
                self.path_crate(def_id.krate)
            }

            DefPathData::Impl => {
                let generics = self.tcx().generics_of(def_id);
                let mut self_ty = self.tcx().type_of(def_id);
                let mut impl_trait_ref = self.tcx().impl_trait_ref(def_id);
                if substs.len() >= generics.count() {
                    self_ty = self_ty.subst(self.tcx(), substs);
                    impl_trait_ref = impl_trait_ref.subst(self.tcx(), substs);
                }
                self.print_impl_path(def_id, substs, self_ty, impl_trait_ref)
            }

            _ => {
                let parent_def_id = DefId { index: key.parent.unwrap(), ..def_id };

                let mut parent_substs = substs;
                let mut trait_qualify_parent = false;
                if !substs.is_empty() {
                    let generics = self.tcx().generics_of(def_id);
                    parent_substs = &substs[..generics.parent_count.min(substs.len())];

                    match key.disambiguated_data.data {
                        // Closures' own generics are only captures, don't print them.
                        DefPathData::ClosureExpr => {}

                        // If we have any generic arguments to print, we do that
                        // on top of the same path, but without its own generics.
                        _ => {
                            if !generics.params.is_empty() && substs.len() >= generics.count() {
                                let args = self.generic_args_to_print(generics, substs);
                                return self.path_generic_args(
                                    |cx| cx.print_def_path(def_id, parent_substs),
                                    args,
                                );
                            }
                        }
                    }

                    // FIXME(eddyb) try to move this into the parent's printing
                    // logic, instead of doing it when printing the child.
                    trait_qualify_parent = generics.has_self
                        && generics.parent == Some(parent_def_id)
                        && parent_substs.len() == generics.parent_count
                        && self.tcx().generics_of(parent_def_id).parent_count == 0;
                }

                self.path_append(
                    |cx: Self| {
                        if trait_qualify_parent {
                            let trait_ref = ty::TraitRef::new(
                                parent_def_id,
                                cx.tcx().intern_substs(parent_substs),
                            );
                            cx.path_qualified(trait_ref.self_ty(), Some(trait_ref))
                        } else {
                            cx.print_def_path(parent_def_id, parent_substs)
                        }
                    },
                    &key.disambiguated_data,
                )
            }
        }
    }

    fn generic_args_to_print(
        &self,
        generics: &'tcx ty::Generics,
        substs: &'tcx [GenericArg<'tcx>],
    ) -> &'tcx [GenericArg<'tcx>] {
        let mut own_params = generics.parent_count..generics.count();

        // Don't print args for `Self` parameters (of traits).
        if generics.has_self && own_params.start == 0 {
            own_params.start = 1;
        }

        // Don't print args that are the defaults of their respective parameters.
        own_params.end -= generics
            .params
            .iter()
            .rev()
            .take_while(|param| match param.kind {
                ty::GenericParamDefKind::Lifetime => false,
                ty::GenericParamDefKind::Type { has_default, .. } => {
                    has_default
                        && substs[param.index as usize]
                            == GenericArg::from(
                                self.tcx().type_of(param.def_id).subst(self.tcx(), substs),
                            )
                }
                ty::GenericParamDefKind::Const { has_default } => {
                    has_default
                        && substs[param.index as usize]
                            == GenericArg::from(self.tcx().const_param_default(param.def_id))
                }
            })
            .count();

        &substs[own_params]
    }

    fn default_print_impl_path(
        self,
        impl_def_id: DefId,
        _substs: &'tcx [GenericArg<'tcx>],
        self_ty: Ty<'tcx>,
        impl_trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        debug!(
            "default_print_impl_path: impl_def_id={:?}, self_ty={}, impl_trait_ref={:?}",
            impl_def_id, self_ty, impl_trait_ref
        );

        let key = self.tcx().def_key(impl_def_id);
        let parent_def_id = DefId { index: key.parent.unwrap(), ..impl_def_id };

        // Decide whether to print the parent path for the impl.
        // Logically, since impls are global, it's never needed, but
        // users may find it useful. Currently, we omit the parent if
        // the impl is either in the same module as the self-type or
        // as the trait.
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
                |cx| cx.print_def_path(parent_def_id, &[]),
                &key.disambiguated_data,
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
///
/// Visited set is needed to avoid full iteration over
/// deeply nested tuples that have no DefId.
fn characteristic_def_id_of_type_cached<'a>(
    ty: Ty<'a>,
    visited: &mut SsoHashSet<Ty<'a>>,
) -> Option<DefId> {
    match *ty.kind() {
        ty::Adt(adt_def, _) => Some(adt_def.did),

        ty::Dynamic(data, ..) => data.principal_def_id(),

        ty::Array(subty, _) | ty::Slice(subty) => {
            characteristic_def_id_of_type_cached(subty, visited)
        }

        ty::RawPtr(mt) => characteristic_def_id_of_type_cached(mt.ty, visited),

        ty::Ref(_, ty, _) => characteristic_def_id_of_type_cached(ty, visited),

        ty::Tuple(ref tys) => tys.iter().find_map(|ty| {
            let ty = ty.expect_ty();
            if visited.insert(ty) {
                return characteristic_def_id_of_type_cached(ty, visited);
            }
            return None;
        }),

        ty::FnDef(def_id, _)
        | ty::Closure(def_id, _)
        | ty::Generator(def_id, _, _)
        | ty::Foreign(def_id) => Some(def_id),

        ty::Bool
        | ty::Char
        | ty::Int(_)
        | ty::Uint(_)
        | ty::Str
        | ty::FnPtr(_)
        | ty::Projection(_)
        | ty::Placeholder(..)
        | ty::Param(_)
        | ty::Opaque(..)
        | ty::Infer(_)
        | ty::Bound(..)
        | ty::Error(_)
        | ty::GeneratorWitness(..)
        | ty::Never
        | ty::Float(_) => None,
    }
}
pub fn characteristic_def_id_of_type(ty: Ty<'_>) -> Option<DefId> {
    characteristic_def_id_of_type_cached(ty, &mut SsoHashSet::new())
}

impl<'tcx, P: Printer<'tcx>> Print<'tcx, P> for ty::RegionKind {
    type Output = P::Region;
    type Error = P::Error;
    fn print(&self, cx: P) -> Result<Self::Output, Self::Error> {
        cx.print_region(self)
    }
}

impl<'tcx, P: Printer<'tcx>> Print<'tcx, P> for ty::Region<'_> {
    type Output = P::Region;
    type Error = P::Error;
    fn print(&self, cx: P) -> Result<Self::Output, Self::Error> {
        cx.print_region(self)
    }
}

impl<'tcx, P: Printer<'tcx>> Print<'tcx, P> for Ty<'tcx> {
    type Output = P::Type;
    type Error = P::Error;
    fn print(&self, cx: P) -> Result<Self::Output, Self::Error> {
        cx.print_type(self)
    }
}

impl<'tcx, P: Printer<'tcx>> Print<'tcx, P>
    for &'tcx ty::List<ty::Binder<'tcx, ty::ExistentialPredicate<'tcx>>>
{
    type Output = P::DynExistential;
    type Error = P::Error;
    fn print(&self, cx: P) -> Result<Self::Output, Self::Error> {
        cx.print_dyn_existential(self)
    }
}

impl<'tcx, P: Printer<'tcx>> Print<'tcx, P> for &'tcx ty::Const<'tcx> {
    type Output = P::Const;
    type Error = P::Error;
    fn print(&self, cx: P) -> Result<Self::Output, Self::Error> {
        cx.print_const(self)
    }
}
