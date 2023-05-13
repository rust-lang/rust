use rustc_data_structures::fx::FxHashSet;
use rustc_errors::ErrorGuaranteed;
use rustc_hir::{def::DefKind, def_id::LocalDefId};
use rustc_middle::ty::util::{CheckRegions, NotUniqueParam};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::ty::{TypeSuperVisitable, TypeVisitable, TypeVisitor};
use rustc_span::Span;
use rustc_type_ir::AliasKind;
use std::ops::ControlFlow;

use crate::errors::{DuplicateArg, NotParam};

struct OpaqueTypeCollector<'tcx> {
    tcx: TyCtxt<'tcx>,
    opaques: Vec<LocalDefId>,
    /// The `DefId` of the item which we are collecting opaque types for.
    item: LocalDefId,

    /// Avoid infinite recursion due to recursive declarations.
    seen: FxHashSet<LocalDefId>,
}

impl<'tcx> OpaqueTypeCollector<'tcx> {
    fn collect(
        tcx: TyCtxt<'tcx>,
        item: LocalDefId,
        val: ty::Binder<'tcx, impl TypeVisitable<TyCtxt<'tcx>>>,
    ) -> Vec<LocalDefId> {
        let mut collector = Self { tcx, opaques: Vec::new(), item, seen: Default::default() };
        val.skip_binder().visit_with(&mut collector);
        collector.opaques
    }

    fn span(&self) -> Span {
        self.tcx.def_span(self.item)
    }

    fn parent(&self) -> Option<LocalDefId> {
        match self.tcx.def_kind(self.item) {
            DefKind::Fn => None,
            DefKind::AssocFn | DefKind::AssocTy | DefKind::AssocConst => {
                Some(self.tcx.local_parent(self.item))
            }
            other => span_bug!(
                self.tcx.def_span(self.item),
                "unhandled item with opaque types: {other:?}"
            ),
        }
    }
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for OpaqueTypeCollector<'tcx> {
    type BreakTy = ErrorGuaranteed;

    #[instrument(skip(self), ret, level = "trace")]
    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<ErrorGuaranteed> {
        match t.kind() {
            ty::Alias(AliasKind::Opaque, alias_ty) if alias_ty.def_id.is_local() => {
                if !self.seen.insert(alias_ty.def_id.expect_local()) {
                    return ControlFlow::Continue(());
                }
                match self.tcx.uses_unique_generic_params(alias_ty.substs, CheckRegions::Bound) {
                    Ok(()) => {
                        // FIXME: implement higher kinded lifetime bounds on nested opaque types. They are not
                        // supported at all, so this is sound to do, but once we want to support them, you'll
                        // start seeing the error below.

                        self.opaques.push(alias_ty.def_id.expect_local());

                        // Collect opaque types nested within the associated type bounds of this opaque type.
                        for (pred, _span) in self
                            .tcx
                            .explicit_item_bounds(alias_ty.def_id)
                            .subst_iter_copied(self.tcx, alias_ty.substs)
                        {
                            trace!(?pred);
                            pred.visit_with(self)?;
                        }

                        ControlFlow::Continue(())
                    }
                    Err(NotUniqueParam::NotParam(arg)) => {
                        let err = self.tcx.sess.emit_err(NotParam {
                            arg,
                            span: self.span(),
                            opaque_span: self.tcx.def_span(alias_ty.def_id),
                        });
                        ControlFlow::Break(err)
                    }
                    Err(NotUniqueParam::DuplicateParam(arg)) => {
                        let err = self.tcx.sess.emit_err(DuplicateArg {
                            arg,
                            span: self.span(),
                            opaque_span: self.tcx.def_span(alias_ty.def_id),
                        });
                        ControlFlow::Break(err)
                    }
                }
            }
            ty::Alias(AliasKind::Projection, alias_ty) => {
                if let Some(parent) = self.parent() {
                    trace!(?alias_ty);
                    let (trait_ref, own_substs) = alias_ty.trait_ref_and_own_substs(self.tcx);

                    trace!(?trait_ref, ?own_substs);
                    // This avoids having to do normalization of `Self::AssocTy` by only
                    // supporting the case of a method defining opaque types from assoc types
                    // in the same impl block.
                    if trait_ref.self_ty() == self.tcx.type_of(parent).subst_identity() {
                        for assoc in self.tcx.associated_items(parent).in_definition_order() {
                            trace!(?assoc);
                            if assoc.trait_item_def_id == Some(alias_ty.def_id) {
                                // We reconstruct the generic args of the associated type within the impl
                                // from the impl's generics and the generic args passed to the type via the
                                // projection.
                                let substs = ty::InternalSubsts::identity_for_item(
                                    self.tcx,
                                    parent.to_def_id(),
                                );
                                trace!(?substs);
                                let substs: Vec<_> =
                                    substs.iter().chain(own_substs.iter().copied()).collect();
                                trace!(?substs);
                                // Find opaque types in this associated type.
                                return self
                                    .tcx
                                    .type_of(assoc.def_id)
                                    .subst(self.tcx, &substs)
                                    .visit_with(self);
                            }
                        }
                    }
                }
                t.super_visit_with(self)
            }
            _ => t.super_visit_with(self),
        }
    }
}

fn opaque_types_defined_by<'tcx>(tcx: TyCtxt<'tcx>, item: LocalDefId) -> &'tcx [LocalDefId] {
    let kind = tcx.def_kind(item);
    trace!(?kind);
    // FIXME(type_alias_impl_trait): This is definitely still wrong except for RPIT and impl trait in assoc types.
    match kind {
        // We're also doing this for `AssocTy` for the wf checks in `check_opaque_meets_bounds`
        DefKind::Fn | DefKind::AssocFn | DefKind::AssocTy | DefKind::AssocConst => {
            let defined_opaques = match kind {
                DefKind::Fn => {
                    OpaqueTypeCollector::collect(tcx, item, tcx.fn_sig(item).subst_identity())
                }
                DefKind::AssocFn => {
                    OpaqueTypeCollector::collect(tcx, item, tcx.fn_sig(item).subst_identity())
                }
                DefKind::AssocTy | DefKind::AssocConst => OpaqueTypeCollector::collect(
                    tcx,
                    item,
                    ty::Binder::dummy(tcx.type_of(item).subst_identity()),
                ),
                _ => unreachable!(),
            };
            tcx.arena.alloc_from_iter(defined_opaques)
        }
        DefKind::Mod
        | DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::Variant
        | DefKind::Trait
        | DefKind::TyAlias
        | DefKind::ForeignTy
        | DefKind::TraitAlias
        | DefKind::TyParam
        | DefKind::Const
        | DefKind::ConstParam
        | DefKind::Static(_)
        | DefKind::Ctor(_, _)
        | DefKind::Macro(_)
        | DefKind::ExternCrate
        | DefKind::Use
        | DefKind::ForeignMod
        | DefKind::AnonConst
        | DefKind::InlineConst
        | DefKind::OpaqueTy
        | DefKind::ImplTraitPlaceholder
        | DefKind::Field
        | DefKind::LifetimeParam
        | DefKind::GlobalAsm
        | DefKind::Impl { .. }
        | DefKind::Closure
        | DefKind::Generator => &[],
    }
}

pub(super) fn provide(providers: &mut ty::query::Providers) {
    *providers = ty::query::Providers { opaque_types_defined_by, ..*providers };
}
