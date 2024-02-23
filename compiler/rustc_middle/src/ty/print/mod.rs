use crate::ty::GenericArg;
use crate::ty::{self, Ty, TyCtxt};

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sso::SsoHashSet;
use rustc_hir as hir;
use rustc_hir::def_id::{CrateNum, DefId, LocalDefId};
use rustc_hir::definitions::{DefPathData, DisambiguatedDefPathData};

// `pretty` is a separate module only for organization.
mod pretty;
pub use self::pretty::*;

pub type PrintError = std::fmt::Error;

pub trait Print<'tcx, P> {
    fn print(&self, cx: &mut P) -> Result<(), PrintError>;
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
    fn tcx<'a>(&'a self) -> TyCtxt<'tcx>;

    fn print_def_path(
        &mut self,
        def_id: DefId,
        args: &'tcx [GenericArg<'tcx>],
    ) -> Result<(), PrintError> {
        self.default_print_def_path(def_id, args)
    }

    fn print_impl_path(
        &mut self,
        impl_def_id: DefId,
        args: &'tcx [GenericArg<'tcx>],
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<(), PrintError> {
        self.default_print_impl_path(impl_def_id, args, self_ty, trait_ref)
    }

    fn print_region(&mut self, region: ty::Region<'tcx>) -> Result<(), PrintError>;

    fn print_type(&mut self, ty: Ty<'tcx>) -> Result<(), PrintError>;

    fn print_dyn_existential(
        &mut self,
        predicates: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    ) -> Result<(), PrintError>;

    fn print_const(&mut self, ct: ty::Const<'tcx>) -> Result<(), PrintError>;

    fn path_crate(&mut self, cnum: CrateNum) -> Result<(), PrintError>;

    fn path_qualified(
        &mut self,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<(), PrintError>;

    fn path_append_impl(
        &mut self,
        print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
        disambiguated_data: &DisambiguatedDefPathData,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<(), PrintError>;

    fn path_append(
        &mut self,
        print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
        disambiguated_data: &DisambiguatedDefPathData,
    ) -> Result<(), PrintError>;

    fn path_generic_args(
        &mut self,
        print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
        args: &[GenericArg<'tcx>],
    ) -> Result<(), PrintError>;

    // Defaults (should not be overridden):

    #[instrument(skip(self), level = "debug")]
    fn default_print_def_path(
        &mut self,
        def_id: DefId,
        args: &'tcx [GenericArg<'tcx>],
    ) -> Result<(), PrintError> {
        let key = self.tcx().def_key(def_id);
        debug!(?key);

        match key.disambiguated_data.data {
            DefPathData::CrateRoot => {
                assert!(key.parent.is_none());
                self.path_crate(def_id.krate)
            }

            DefPathData::Impl => {
                let generics = self.tcx().generics_of(def_id);
                let self_ty = self.tcx().type_of(def_id);
                let impl_trait_ref = self.tcx().impl_trait_ref(def_id);
                let (self_ty, impl_trait_ref) = if args.len() >= generics.count() {
                    (
                        self_ty.instantiate(self.tcx(), args),
                        impl_trait_ref.map(|i| i.instantiate(self.tcx(), args)),
                    )
                } else {
                    (
                        self_ty.instantiate_identity(),
                        impl_trait_ref.map(|i| i.instantiate_identity()),
                    )
                };
                self.print_impl_path(def_id, args, self_ty, impl_trait_ref)
            }

            _ => {
                let parent_def_id = DefId { index: key.parent.unwrap(), ..def_id };

                let mut parent_args = args;
                let mut trait_qualify_parent = false;
                if !args.is_empty() {
                    let generics = self.tcx().generics_of(def_id);
                    parent_args = &args[..generics.parent_count.min(args.len())];

                    match key.disambiguated_data.data {
                        DefPathData::Closure => {
                            // FIXME(async_closures): This is somewhat ugly.
                            // We need to additionally print the `kind` field of a closure if
                            // it is desugared from a coroutine-closure.
                            if let Some(hir::CoroutineKind::Desugared(
                                _,
                                hir::CoroutineSource::Closure,
                            )) = self.tcx().coroutine_kind(def_id)
                                && args.len() > parent_args.len()
                            {
                                return self.path_generic_args(
                                    |cx| cx.print_def_path(def_id, parent_args),
                                    &args[..parent_args.len() + 1][..1],
                                );
                            } else {
                                // Closures' own generics are only captures, don't print them.
                            }
                        }
                        // This covers both `DefKind::AnonConst` and `DefKind::InlineConst`.
                        // Anon consts doesn't have their own generics, and inline consts' own
                        // generics are their inferred types, so don't print them.
                        DefPathData::AnonConst => {}

                        // If we have any generic arguments to print, we do that
                        // on top of the same path, but without its own generics.
                        _ => {
                            if !generics.params.is_empty() && args.len() >= generics.count() {
                                let args = generics.own_args_no_defaults(self.tcx(), args);
                                return self.path_generic_args(
                                    |cx| cx.print_def_path(def_id, parent_args),
                                    args,
                                );
                            }
                        }
                    }

                    // FIXME(eddyb) try to move this into the parent's printing
                    // logic, instead of doing it when printing the child.
                    trait_qualify_parent = generics.has_self
                        && generics.parent == Some(parent_def_id)
                        && parent_args.len() == generics.parent_count
                        && self.tcx().generics_of(parent_def_id).parent_count == 0;
                }

                self.path_append(
                    |cx: &mut Self| {
                        if trait_qualify_parent {
                            let trait_ref = ty::TraitRef::new(
                                cx.tcx(),
                                parent_def_id,
                                parent_args.iter().copied(),
                            );
                            cx.path_qualified(trait_ref.self_ty(), Some(trait_ref))
                        } else {
                            cx.print_def_path(parent_def_id, parent_args)
                        }
                    },
                    &key.disambiguated_data,
                )
            }
        }
    }

    fn default_print_impl_path(
        &mut self,
        impl_def_id: DefId,
        _args: &'tcx [GenericArg<'tcx>],
        self_ty: Ty<'tcx>,
        impl_trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<(), PrintError> {
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
            Some(ty_def_id) => self.tcx().parent(ty_def_id) == parent_def_id,
        };
        let in_trait_mod = match impl_trait_ref {
            None => false,
            Some(trait_ref) => self.tcx().parent(trait_ref.def_id) == parent_def_id,
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
        ty::Adt(adt_def, _) => Some(adt_def.did()),

        ty::Dynamic(data, ..) => data.principal_def_id(),

        ty::Array(subty, _) | ty::Slice(subty) => {
            characteristic_def_id_of_type_cached(subty, visited)
        }

        ty::RawPtr(mt) => characteristic_def_id_of_type_cached(mt.ty, visited),

        ty::Ref(_, ty, _) => characteristic_def_id_of_type_cached(ty, visited),

        ty::Tuple(tys) => tys.iter().find_map(|ty| {
            if visited.insert(ty) {
                return characteristic_def_id_of_type_cached(ty, visited);
            }
            return None;
        }),

        ty::FnDef(def_id, _)
        | ty::Closure(def_id, _)
        | ty::CoroutineClosure(def_id, _)
        | ty::Coroutine(def_id, _)
        | ty::CoroutineWitness(def_id, _)
        | ty::Foreign(def_id) => Some(def_id),

        ty::Bool
        | ty::Char
        | ty::Int(_)
        | ty::Uint(_)
        | ty::Str
        | ty::FnPtr(_)
        | ty::Alias(..)
        | ty::Placeholder(..)
        | ty::Param(_)
        | ty::Infer(_)
        | ty::Bound(..)
        | ty::Error(_)
        | ty::Never
        | ty::Float(_) => None,
    }
}
pub fn characteristic_def_id_of_type(ty: Ty<'_>) -> Option<DefId> {
    characteristic_def_id_of_type_cached(ty, &mut SsoHashSet::new())
}

impl<'tcx, P: Printer<'tcx>> Print<'tcx, P> for ty::Region<'tcx> {
    fn print(&self, cx: &mut P) -> Result<(), PrintError> {
        cx.print_region(*self)
    }
}

impl<'tcx, P: Printer<'tcx>> Print<'tcx, P> for Ty<'tcx> {
    fn print(&self, cx: &mut P) -> Result<(), PrintError> {
        cx.print_type(*self)
    }
}

impl<'tcx, P: Printer<'tcx>> Print<'tcx, P> for &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>> {
    fn print(&self, cx: &mut P) -> Result<(), PrintError> {
        cx.print_dyn_existential(self)
    }
}

impl<'tcx, P: Printer<'tcx>> Print<'tcx, P> for ty::Const<'tcx> {
    fn print(&self, cx: &mut P) -> Result<(), PrintError> {
        cx.print_const(*self)
    }
}

// This is only used by query descriptions
pub fn describe_as_module(def_id: impl Into<LocalDefId>, tcx: TyCtxt<'_>) -> String {
    let def_id = def_id.into();
    if def_id.is_top_level_module() {
        "top-level module".to_string()
    } else {
        format!("module `{}`", tcx.def_path_str(def_id))
    }
}
