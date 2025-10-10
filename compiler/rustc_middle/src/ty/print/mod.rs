use hir::def::Namespace;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sso::SsoHashSet;
use rustc_hir as hir;
use rustc_hir::def_id::{CrateNum, DefId};
use rustc_hir::definitions::{DefPathData, DisambiguatedDefPathData};
use tracing::{debug, instrument, trace};

use crate::ty::{self, GenericArg, Ty, TyCtxt};

// `pretty` is a separate module only for organization.
mod pretty;
pub use self::pretty::*;
use super::Lift;

pub type PrintError = std::fmt::Error;

pub trait Print<'tcx, P> {
    fn print(&self, p: &mut P) -> Result<(), PrintError>;
}

/// A trait that "prints" user-facing type system entities: paths, types, lifetimes, constants,
/// etc. "Printing" here means building up a representation of the entity's path, usually as a
/// `String` (e.g. "std::io::Read") or a `Vec<Symbol>` (e.g. `[sym::std, sym::io, sym::Read]`). The
/// representation is built up by appending one or more pieces. The specific details included in
/// the built-up representation depend on the purpose of the printer. The more advanced printers
/// also rely on the `PrettyPrinter` sub-trait.
pub trait Printer<'tcx>: Sized {
    fn tcx<'a>(&'a self) -> TyCtxt<'tcx>;

    /// Appends a representation of an entity with a normal path, e.g. "std::io::Read".
    fn print_def_path(
        &mut self,
        def_id: DefId,
        args: &'tcx [GenericArg<'tcx>],
    ) -> Result<(), PrintError> {
        self.default_print_def_path(def_id, args)
    }

    /// Like `print_def_path`, but for `DefPathData::Impl`.
    fn print_impl_path(
        &mut self,
        impl_def_id: DefId,
        args: &'tcx [GenericArg<'tcx>],
    ) -> Result<(), PrintError> {
        let tcx = self.tcx();
        let self_ty = tcx.type_of(impl_def_id);
        let impl_trait_ref = tcx.impl_trait_ref(impl_def_id);
        let (self_ty, impl_trait_ref) = if tcx.generics_of(impl_def_id).count() <= args.len() {
            (
                self_ty.instantiate(tcx, args),
                impl_trait_ref.map(|impl_trait_ref| impl_trait_ref.instantiate(tcx, args)),
            )
        } else {
            // We are probably printing a nested item inside of an impl.
            // Use the identity substitutions for the impl.
            (
                self_ty.instantiate_identity(),
                impl_trait_ref.map(|impl_trait_ref| impl_trait_ref.instantiate_identity()),
            )
        };

        self.default_print_impl_path(impl_def_id, self_ty, impl_trait_ref)
    }

    /// Appends a representation of a region.
    fn print_region(&mut self, region: ty::Region<'tcx>) -> Result<(), PrintError>;

    /// Appends a representation of a type.
    fn print_type(&mut self, ty: Ty<'tcx>) -> Result<(), PrintError>;

    /// Appends a representation of a list of `PolyExistentialPredicate`s.
    fn print_dyn_existential(
        &mut self,
        predicates: &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>>,
    ) -> Result<(), PrintError>;

    /// Appends a representation of a const.
    fn print_const(&mut self, ct: ty::Const<'tcx>) -> Result<(), PrintError>;

    /// Appends a representation of a crate name, e.g. `std`, or even ``.
    fn print_crate_name(&mut self, cnum: CrateNum) -> Result<(), PrintError>;

    /// Appends a representation of a (full or partial) simple path, in two parts. `print_prefix`,
    /// when called, appends the representation of the leading segments. The rest of the method
    /// appends the representation of the final segment, the details of which are in
    /// `disambiguated_data`.
    ///
    /// E.g. `std::io` + `Read` -> `std::io::Read`.
    fn print_path_with_simple(
        &mut self,
        print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
        disambiguated_data: &DisambiguatedDefPathData,
    ) -> Result<(), PrintError>;

    /// Similar to `print_path_with_simple`, but the final segment is an `impl` segment.
    ///
    /// E.g. `slice` + `<impl [T]>` -> `slice::<impl [T]>`, which may then be further appended to,
    /// giving a longer path representation such as `slice::<impl [T]>::to_vec_in::ConvertVec`.
    fn print_path_with_impl(
        &mut self,
        print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<(), PrintError>;

    /// Appends a representation of a path ending in generic args, in two parts. `print_prefix`,
    /// when called, appends the leading segments. The rest of the method appends the
    /// representation of the generic args. (Some printers choose to skip appending the generic
    /// args.)
    ///
    /// E.g. `ImplementsTraitForUsize` + `<usize>` -> `ImplementsTraitForUsize<usize>`.
    fn print_path_with_generic_args(
        &mut self,
        print_prefix: impl FnOnce(&mut Self) -> Result<(), PrintError>,
        args: &[GenericArg<'tcx>],
    ) -> Result<(), PrintError>;

    /// Appends a representation of a qualified path segment, e.g. `<OsString as From<&T>>`.
    /// If `trait_ref` is `None`, it may fall back to simpler forms, e.g. `<Vec<T>>` or just `Foo`.
    fn print_path_with_qualified(
        &mut self,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<(), PrintError>;

    fn print_coroutine_with_kind(
        &mut self,
        def_id: DefId,
        parent_args: &'tcx [GenericArg<'tcx>],
        kind: Ty<'tcx>,
    ) -> Result<(), PrintError> {
        self.print_path_with_generic_args(|p| p.print_def_path(def_id, parent_args), &[kind.into()])
    }

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
                self.print_crate_name(def_id.krate)
            }

            DefPathData::Impl => self.print_impl_path(def_id, args),

            _ => {
                let parent_def_id = DefId { index: key.parent.unwrap(), ..def_id };

                let mut parent_args = args;
                let mut trait_qualify_parent = false;
                if !args.is_empty() {
                    let generics = self.tcx().generics_of(def_id);
                    parent_args = &args[..generics.parent_count.min(args.len())];

                    match key.disambiguated_data.data {
                        DefPathData::Closure => {
                            // We need to additionally print the `kind` field of a coroutine if
                            // it is desugared from a coroutine-closure.
                            if let Some(hir::CoroutineKind::Desugared(
                                _,
                                hir::CoroutineSource::Closure,
                            )) = self.tcx().coroutine_kind(def_id)
                                && args.len() > parent_args.len()
                            {
                                return self.print_coroutine_with_kind(
                                    def_id,
                                    parent_args,
                                    args[parent_args.len()].expect_ty(),
                                );
                            } else {
                                // Closures' own generics are only captures, don't print them.
                            }
                        }
                        DefPathData::SyntheticCoroutineBody => {
                            // Synthetic coroutine bodies have no distinct generics, since like
                            // closures they're all just internal state of the coroutine.
                        }
                        // This covers both `DefKind::AnonConst` and `DefKind::InlineConst`.
                        // Anon consts doesn't have their own generics, and inline consts' own
                        // generics are their inferred types, so don't print them.
                        DefPathData::AnonConst => {}

                        // If we have any generic arguments to print, we do that
                        // on top of the same path, but without its own generics.
                        _ => {
                            if !generics.is_own_empty() && args.len() >= generics.count() {
                                let args = generics.own_args_no_defaults(self.tcx(), args);
                                return self.print_path_with_generic_args(
                                    |p| p.print_def_path(def_id, parent_args),
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

                self.print_path_with_simple(
                    |p: &mut Self| {
                        if trait_qualify_parent {
                            let trait_ref = ty::TraitRef::new(
                                p.tcx(),
                                parent_def_id,
                                parent_args.iter().copied(),
                            );
                            p.print_path_with_qualified(trait_ref.self_ty(), Some(trait_ref))
                        } else {
                            p.print_def_path(parent_def_id, parent_args)
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
            self.print_path_with_impl(
                |p| p.print_def_path(parent_def_id, &[]),
                self_ty,
                impl_trait_ref,
            )
        } else {
            // Otherwise, try to give a good form that would be valid language
            // syntax. Preferably using associated item notation.
            self.print_path_with_qualified(self_ty, impl_trait_ref)
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

        ty::Pat(subty, _) | ty::Array(subty, _) | ty::Slice(subty) => {
            characteristic_def_id_of_type_cached(subty, visited)
        }

        ty::RawPtr(ty, _) => characteristic_def_id_of_type_cached(ty, visited),

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
        | ty::FnPtr(..)
        | ty::UnsafeBinder(_)
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
    fn print(&self, p: &mut P) -> Result<(), PrintError> {
        p.print_region(*self)
    }
}

impl<'tcx, P: Printer<'tcx>> Print<'tcx, P> for Ty<'tcx> {
    fn print(&self, p: &mut P) -> Result<(), PrintError> {
        p.print_type(*self)
    }
}

impl<'tcx, P: Printer<'tcx> + std::fmt::Write> Print<'tcx, P> for ty::Instance<'tcx> {
    fn print(&self, cx: &mut P) -> Result<(), PrintError> {
        cx.print_def_path(self.def_id(), self.args)?;
        match self.def {
            ty::InstanceKind::Item(_) => {}
            ty::InstanceKind::VTableShim(_) => cx.write_str(" - shim(vtable)")?,
            ty::InstanceKind::ReifyShim(_, None) => cx.write_str(" - shim(reify)")?,
            ty::InstanceKind::ReifyShim(_, Some(ty::ReifyReason::FnPtr)) => {
                cx.write_str(" - shim(reify-fnptr)")?
            }
            ty::InstanceKind::ReifyShim(_, Some(ty::ReifyReason::Vtable)) => {
                cx.write_str(" - shim(reify-vtable)")?
            }
            ty::InstanceKind::ThreadLocalShim(_) => cx.write_str(" - shim(tls)")?,
            ty::InstanceKind::Intrinsic(_) => cx.write_str(" - intrinsic")?,
            ty::InstanceKind::Virtual(_, num) => cx.write_str(&format!(" - virtual#{num}"))?,
            ty::InstanceKind::FnPtrShim(_, ty) => cx.write_str(&format!(" - shim({ty})"))?,
            ty::InstanceKind::ClosureOnceShim { .. } => cx.write_str(" - shim")?,
            ty::InstanceKind::ConstructCoroutineInClosureShim { .. } => cx.write_str(" - shim")?,
            ty::InstanceKind::DropGlue(_, None) => cx.write_str(" - shim(None)")?,
            ty::InstanceKind::DropGlue(_, Some(ty)) => {
                cx.write_str(&format!(" - shim(Some({ty}))"))?
            }
            ty::InstanceKind::CloneShim(_, ty) => cx.write_str(&format!(" - shim({ty})"))?,
            ty::InstanceKind::FnPtrAddrShim(_, ty) => cx.write_str(&format!(" - shim({ty})"))?,
            ty::InstanceKind::FutureDropPollShim(_, proxy_ty, impl_ty) => {
                cx.write_str(&format!(" - dropshim({proxy_ty}-{impl_ty})"))?
            }
            ty::InstanceKind::AsyncDropGlue(_, ty) => cx.write_str(&format!(" - shim({ty})"))?,
            ty::InstanceKind::AsyncDropGlueCtorShim(_, ty) => {
                cx.write_str(&format!(" - shim(Some({ty}))"))?
            }
        };
        Ok(())
    }
}

impl<'tcx, P: Printer<'tcx>> Print<'tcx, P> for &'tcx ty::List<ty::PolyExistentialPredicate<'tcx>> {
    fn print(&self, p: &mut P) -> Result<(), PrintError> {
        p.print_dyn_existential(self)
    }
}

impl<'tcx, P: Printer<'tcx>> Print<'tcx, P> for ty::Const<'tcx> {
    fn print(&self, p: &mut P) -> Result<(), PrintError> {
        p.print_const(*self)
    }
}

impl<T> rustc_type_ir::ir_print::IrPrint<T> for TyCtxt<'_>
where
    T: Copy + for<'a, 'tcx> Lift<TyCtxt<'tcx>, Lifted: Print<'tcx, FmtPrinter<'a, 'tcx>>>,
{
    fn print(t: &T, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        ty::tls::with(|tcx| {
            let mut p = FmtPrinter::new(tcx, Namespace::TypeNS);
            tcx.lift(*t).expect("could not lift for printing").print(&mut p)?;
            fmt.write_str(&p.into_buffer())?;
            Ok(())
        })
    }

    fn print_debug(t: &T, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        with_no_trimmed_paths!(Self::print(t, fmt))
    }
}
