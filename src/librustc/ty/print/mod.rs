use crate::hir::map::DefPathData;
use crate::hir::def_id::{CrateNum, DefId};
use crate::ty::{self, DefIdTree, Ty, TyCtxt, TypeFoldable};
use crate::ty::subst::{Subst, SubstsRef};

use rustc_data_structures::fx::FxHashSet;
use syntax::symbol::InternedString;

use std::iter;
use std::ops::Deref;

// `pretty` is a separate module only for organization.
mod pretty;
pub use self::pretty::*;

// FIXME(eddyb) this module uses `pub(crate)` for things used only
// from `ppaux` - when that is removed, they can be re-privatized.

struct LateBoundRegionNameCollector(FxHashSet<InternedString>);
impl<'tcx> ty::fold::TypeVisitor<'tcx> for LateBoundRegionNameCollector {
    fn visit_region(&mut self, r: ty::Region<'tcx>) -> bool {
        match *r {
            ty::ReLateBound(_, ty::BrNamed(_, name)) => {
                self.0.insert(name);
            },
            _ => {},
        }
        r.super_visit_with(self)
    }
}

pub(crate) struct PrintConfig {
    pub(crate) is_debug: bool,
    pub(crate) is_verbose: bool,
    pub(crate) identify_regions: bool,
    used_region_names: Option<FxHashSet<InternedString>>,
    region_index: usize,
    binder_depth: usize,
}

impl PrintConfig {
    fn new(tcx: TyCtxt<'_, '_, '_>) -> Self {
        PrintConfig {
            is_debug: false,
            is_verbose: tcx.sess.verbose(),
            identify_regions: tcx.sess.opts.debugging_opts.identify_regions,
            used_region_names: None,
            region_index: 0,
            binder_depth: 0,
        }
    }
}

pub struct PrintCx<'a, 'gcx, 'tcx, P> {
    pub tcx: TyCtxt<'a, 'gcx, 'tcx>,
    pub printer: P,
    pub(crate) config: &'a mut PrintConfig,
}

// HACK(eddyb) this is solely for `self: PrintCx<Self>`, e.g. to
// implement traits on the printer and call the methods on the context.
impl<P> Deref for PrintCx<'_, '_, '_, P> {
    type Target = P;
    fn deref(&self) -> &P {
        &self.printer
    }
}

impl<'a, 'gcx, 'tcx, P> PrintCx<'a, 'gcx, 'tcx, P> {
    pub fn with<R>(
        tcx: TyCtxt<'a, 'gcx, 'tcx>,
        printer: P,
        f: impl FnOnce(PrintCx<'_, 'gcx, 'tcx, P>) -> R,
    ) -> R {
        f(PrintCx {
            tcx,
            printer,
            config: &mut PrintConfig::new(tcx),
        })
    }

    pub(crate) fn with_tls_tcx<R>(printer: P, f: impl FnOnce(PrintCx<'_, '_, '_, P>) -> R) -> R {
        ty::tls::with(|tcx| PrintCx::with(tcx, printer, f))
    }
    fn prepare_late_bound_region_info<T>(&mut self, value: &ty::Binder<T>)
    where T: TypeFoldable<'tcx>
    {
        let mut collector = LateBoundRegionNameCollector(Default::default());
        value.visit_with(&mut collector);
        self.config.used_region_names = Some(collector.0);
        self.config.region_index = 0;
    }
}

pub trait Print<'tcx, P> {
    type Output;
    type Error;

    fn print(&self, cx: PrintCx<'_, '_, 'tcx, P>) -> Result<Self::Output, Self::Error>;
    fn print_display(
        &self,
        cx: PrintCx<'_, '_, 'tcx, P>,
    ) -> Result<Self::Output, Self::Error> {
        let old_debug = cx.config.is_debug;
        cx.config.is_debug = false;
        let result = self.print(PrintCx {
            tcx: cx.tcx,
            printer: cx.printer,
            config: cx.config,
        });
        cx.config.is_debug = old_debug;
        result
    }
    fn print_debug(&self, cx: PrintCx<'_, '_, 'tcx, P>) -> Result<Self::Output, Self::Error> {
        let old_debug = cx.config.is_debug;
        cx.config.is_debug = true;
        let result = self.print(PrintCx {
            tcx: cx.tcx,
            printer: cx.printer,
            config: cx.config,
        });
        cx.config.is_debug = old_debug;
        result
    }
}

pub trait Printer: Sized {
    type Error;

    type Path;
    type Region;
    type Type;

    fn print_def_path(
        self: PrintCx<'_, '_, 'tcx, Self>,
        def_id: DefId,
        substs: Option<SubstsRef<'tcx>>,
        projections: impl Iterator<Item = ty::ExistentialProjection<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        self.default_print_def_path(def_id, substs, projections)
    }
    fn print_impl_path(
        self: PrintCx<'_, '_, 'tcx, Self>,
        impl_def_id: DefId,
        substs: Option<SubstsRef<'tcx>>,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error> {
        self.default_print_impl_path(impl_def_id, substs, self_ty, trait_ref)
    }

    fn print_region(
        self: PrintCx<'_, '_, '_, Self>,
        region: ty::Region<'_>,
    ) -> Result<Self::Region, Self::Error>;

    fn print_type(
        self: PrintCx<'_, '_, 'tcx, Self>,
        ty: Ty<'tcx>,
    ) -> Result<Self::Type, Self::Error>;

    fn path_crate(
        self: PrintCx<'_, '_, '_, Self>,
        cnum: CrateNum,
    ) -> Result<Self::Path, Self::Error>;
    fn path_qualified(
        self: PrintCx<'_, '_, 'tcx, Self>,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error>;

    fn path_append_impl<'gcx, 'tcx>(
        self: PrintCx<'_, 'gcx, 'tcx, Self>,
        print_prefix: impl FnOnce(
            PrintCx<'_, 'gcx, 'tcx, Self>,
        ) -> Result<Self::Path, Self::Error>,
        self_ty: Ty<'tcx>,
        trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<Self::Path, Self::Error>;
    fn path_append<'gcx, 'tcx>(
        self: PrintCx<'_, 'gcx, 'tcx, Self>,
        print_prefix: impl FnOnce(
            PrintCx<'_, 'gcx, 'tcx, Self>,
        ) -> Result<Self::Path, Self::Error>,
        text: &str,
    ) -> Result<Self::Path, Self::Error>;
    fn path_generic_args<'gcx, 'tcx>(
        self: PrintCx<'_, 'gcx, 'tcx, Self>,
        print_prefix: impl FnOnce(
            PrintCx<'_, 'gcx, 'tcx, Self>,
        ) -> Result<Self::Path, Self::Error>,
        params: &[ty::GenericParamDef],
        substs: SubstsRef<'tcx>,
        projections: impl Iterator<Item = ty::ExistentialProjection<'tcx>>,
    ) -> Result<Self::Path, Self::Error>;
}

impl<P: Printer> PrintCx<'a, 'gcx, 'tcx, P> {
    pub fn default_print_def_path(
        self,
        def_id: DefId,
        substs: Option<SubstsRef<'tcx>>,
        projections: impl Iterator<Item = ty::ExistentialProjection<'tcx>>,
    ) -> Result<P::Path, P::Error> {
        debug!("default_print_def_path: def_id={:?}, substs={:?}", def_id, substs);
        let key = self.tcx.def_key(def_id);
        debug!("default_print_def_path: key={:?}", key);

        match key.disambiguated_data.data {
            DefPathData::CrateRoot => {
                assert!(key.parent.is_none());
                self.path_crate(def_id.krate)
            }

            DefPathData::Impl => {
                let mut self_ty = self.tcx.type_of(def_id);
                if let Some(substs) = substs {
                    self_ty = self_ty.subst(self.tcx, substs);
                }

                let mut impl_trait_ref = self.tcx.impl_trait_ref(def_id);
                if let Some(substs) = substs {
                    impl_trait_ref = impl_trait_ref.subst(self.tcx, substs);
                }
                self.print_impl_path(def_id, substs, self_ty, impl_trait_ref)
            }

            _ => {
                let generics = substs.map(|_| self.tcx.generics_of(def_id));
                let generics_parent = generics.as_ref().and_then(|g| g.parent);
                let parent_def_id = DefId { index: key.parent.unwrap(), ..def_id };
                let print_parent_path = |cx: PrintCx<'_, 'gcx, 'tcx, P>| {
                    if let Some(generics_parent_def_id) = generics_parent {
                        assert_eq!(parent_def_id, generics_parent_def_id);

                        // FIXME(eddyb) try to move this into the parent's printing
                        // logic, instead of doing it when printing the child.
                        let parent_generics = cx.tcx.generics_of(parent_def_id);
                        let parent_has_own_self =
                            parent_generics.has_self && parent_generics.parent_count == 0;
                        if let (Some(substs), true) = (substs, parent_has_own_self) {
                            let trait_ref = ty::TraitRef::new(parent_def_id, substs);
                            cx.path_qualified(trait_ref.self_ty(), Some(trait_ref))
                        } else {
                            cx.print_def_path(parent_def_id, substs, iter::empty())
                        }
                    } else {
                        cx.print_def_path(parent_def_id, None, iter::empty())
                    }
                };
                let print_path = |cx: PrintCx<'_, 'gcx, 'tcx, P>| {
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
                    let has_own_self = generics.has_self && generics.parent_count == 0;
                    let params = &generics.params[has_own_self as usize..];
                    self.path_generic_args(print_path, params, substs, projections)
                } else {
                    print_path(self)
                }
            }
        }
    }

    fn default_print_impl_path(
        self,
        impl_def_id: DefId,
        _substs: Option<SubstsRef<'tcx>>,
        self_ty: Ty<'tcx>,
        impl_trait_ref: Option<ty::TraitRef<'tcx>>,
    ) -> Result<P::Path, P::Error> {
        debug!("default_print_impl_path: impl_def_id={:?}, self_ty={}, impl_trait_ref={:?}",
               impl_def_id, self_ty, impl_trait_ref);

        // Decide whether to print the parent path for the impl.
        // Logically, since impls are global, it's never needed, but
        // users may find it useful. Currently, we omit the parent if
        // the impl is either in the same module as the self-type or
        // as the trait.
        let parent_def_id = self.tcx.parent(impl_def_id).unwrap();
        let in_self_mod = match characteristic_def_id_of_type(self_ty) {
            None => false,
            Some(ty_def_id) => self.tcx.parent(ty_def_id) == Some(parent_def_id),
        };
        let in_trait_mod = match impl_trait_ref {
            None => false,
            Some(trait_ref) => self.tcx.parent(trait_ref.def_id) == Some(parent_def_id),
        };

        if !in_self_mod && !in_trait_mod {
            // If the impl is not co-located with either self-type or
            // trait-type, then fallback to a format that identifies
            // the module more clearly.
            self.path_append_impl(
                |cx| cx.print_def_path(parent_def_id, None, iter::empty()),
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
