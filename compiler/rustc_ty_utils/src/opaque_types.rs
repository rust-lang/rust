use rustc_data_structures::fx::FxHashSet;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit;
use rustc_hir::intravisit::Visitor;
use rustc_middle::query::Providers;
use rustc_middle::ty::util::{CheckRegions, NotUniqueParam};
use rustc_middle::ty::{self, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitor};
use rustc_middle::{bug, span_bug};
use rustc_span::Span;
use tracing::{instrument, trace};

use crate::errors::{DuplicateArg, NotParam};

struct OpaqueTypeCollector<'tcx> {
    tcx: TyCtxt<'tcx>,
    opaques: Vec<LocalDefId>,
    /// The `DefId` of the item which we are collecting opaque types for.
    item: LocalDefId,

    /// Avoid infinite recursion due to recursive declarations.
    seen: FxHashSet<LocalDefId>,

    span: Option<Span>,

    mode: CollectionMode,
}

enum CollectionMode {
    /// For impl trait in assoc types we only permit collecting them from
    /// associated types of the same impl block.
    ImplTraitInAssocTypes,
    /// When collecting for an explicit `#[define_opaque]` attribute, find all TAITs
    Taits,
    /// The default case, only collect RPITs and AsyncFn return types, as these are
    /// always defined by the current item.
    RpitAndAsyncFnOnly,
}

impl<'tcx> OpaqueTypeCollector<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, item: LocalDefId) -> Self {
        let mode = match tcx.def_kind(item) {
            DefKind::AssocConst | DefKind::AssocFn | DefKind::AssocTy => {
                CollectionMode::ImplTraitInAssocTypes
            }
            DefKind::TyAlias => CollectionMode::Taits,
            _ => CollectionMode::RpitAndAsyncFnOnly,
        };
        Self { tcx, opaques: Vec::new(), item, seen: Default::default(), span: None, mode }
    }

    fn span(&self) -> Span {
        self.span.unwrap_or_else(|| {
            self.tcx.def_ident_span(self.item).unwrap_or_else(|| self.tcx.def_span(self.item))
        })
    }

    fn visit_spanned(&mut self, span: Span, value: impl TypeVisitable<TyCtxt<'tcx>>) {
        let old = self.span;
        self.span = Some(span);
        value.visit_with(self);
        self.span = old;
    }

    fn parent_impl_trait_ref(&self) -> Option<ty::TraitRef<'tcx>> {
        let parent = self.parent()?;
        if matches!(self.tcx.def_kind(parent), DefKind::Impl { .. }) {
            Some(self.tcx.impl_trait_ref(parent)?.instantiate_identity())
        } else {
            None
        }
    }

    fn parent(&self) -> Option<LocalDefId> {
        match self.tcx.def_kind(self.item) {
            DefKind::AssocFn | DefKind::AssocTy | DefKind::AssocConst => {
                Some(self.tcx.local_parent(self.item))
            }
            _ => None,
        }
    }

    #[instrument(level = "trace", skip(self))]
    fn collect_taits_declared_in_body(&mut self) {
        let body = self.tcx.hir_body_owned_by(self.item).value;
        struct TaitInBodyFinder<'a, 'tcx> {
            collector: &'a mut OpaqueTypeCollector<'tcx>,
        }
        impl<'v> intravisit::Visitor<'v> for TaitInBodyFinder<'_, '_> {
            #[instrument(level = "trace", skip(self))]
            fn visit_nested_item(&mut self, id: rustc_hir::ItemId) {
                let id = id.owner_id.def_id;
                if let DefKind::TyAlias = self.collector.tcx.def_kind(id) {
                    let items = self.collector.tcx.opaque_types_defined_by(id);
                    self.collector.opaques.extend(items);
                }
            }
            #[instrument(level = "trace", skip(self))]
            // Recurse into these, as they are type checked with their parent
            fn visit_nested_body(&mut self, id: rustc_hir::BodyId) {
                let body = self.collector.tcx.hir_body(id);
                self.visit_body(body);
            }
        }
        TaitInBodyFinder { collector: self }.visit_expr(body);
    }

    #[instrument(level = "debug", skip(self))]
    fn visit_opaque_ty(&mut self, alias_ty: ty::AliasTy<'tcx>) {
        if !self.seen.insert(alias_ty.def_id.expect_local()) {
            return;
        }

        // TAITs outside their defining scopes are ignored.
        match self.tcx.local_opaque_ty_origin(alias_ty.def_id.expect_local()) {
            rustc_hir::OpaqueTyOrigin::FnReturn { .. }
            | rustc_hir::OpaqueTyOrigin::AsyncFn { .. } => {}
            rustc_hir::OpaqueTyOrigin::TyAlias { in_assoc_ty, .. } => match self.mode {
                // If we are collecting opaques in an assoc method, we are only looking at assoc types
                // mentioned in the assoc method and only at opaques defined in there. We do not
                // want to collect TAITs
                CollectionMode::ImplTraitInAssocTypes => {
                    if !in_assoc_ty {
                        return;
                    }
                }
                // If we are collecting opaques referenced from a `define_opaque` attribute, we
                // do not want to look at opaques defined in associated types. Those can only be
                // defined by methods on the same impl.
                CollectionMode::Taits => {
                    if in_assoc_ty {
                        return;
                    }
                }
                CollectionMode::RpitAndAsyncFnOnly => return,
            },
        }

        trace!(?alias_ty, "adding");
        self.opaques.push(alias_ty.def_id.expect_local());

        let parent_count = self.tcx.generics_of(alias_ty.def_id).parent_count;
        // Only check that the parent generics of the TAIT/RPIT are unique.
        // the args owned by the opaque are going to always be duplicate
        // lifetime params for RPITs, and empty for TAITs.
        match self
            .tcx
            .uses_unique_generic_params(&alias_ty.args[..parent_count], CheckRegions::FromFunction)
        {
            Ok(()) => {
                // FIXME: implement higher kinded lifetime bounds on nested opaque types. They are not
                // supported at all, so this is sound to do, but once we want to support them, you'll
                // start seeing the error below.

                // Collect opaque types nested within the associated type bounds of this opaque type.
                // We use identity args here, because we already know that the opaque type uses
                // only generic parameters, and thus instantiating would not give us more information.
                for (pred, span) in
                    self.tcx.explicit_item_bounds(alias_ty.def_id).iter_identity_copied()
                {
                    trace!(?pred);
                    self.visit_spanned(span, pred);
                }
            }
            Err(NotUniqueParam::NotParam(arg)) => {
                self.tcx.dcx().emit_err(NotParam {
                    arg,
                    span: self.span(),
                    opaque_span: self.tcx.def_span(alias_ty.def_id),
                });
            }
            Err(NotUniqueParam::DuplicateParam(arg)) => {
                self.tcx.dcx().emit_err(DuplicateArg {
                    arg,
                    span: self.span(),
                    opaque_span: self.tcx.def_span(alias_ty.def_id),
                });
            }
        }
    }

    /// Checks the `#[define_opaque]` attributes on items and collects opaques to define
    /// from the referenced types.
    #[instrument(level = "trace", skip(self))]
    fn collect_taits_from_defines_attr(&mut self) {
        let hir_id = self.tcx.local_def_id_to_hir_id(self.item);
        if !hir_id.is_owner() {
            return;
        }
        let Some(defines) = self.tcx.hir_attr_map(hir_id.owner).define_opaque else {
            return;
        };
        for &(span, define) in defines {
            trace!(?define);
            let mode = std::mem::replace(&mut self.mode, CollectionMode::Taits);
            let n = self.opaques.len();
            super::sig_types::walk_types(self.tcx, define, self);
            if n == self.opaques.len() {
                self.tcx.dcx().span_err(span, "item does not contain any opaque types");
            }
            self.mode = mode;
        }
        // Allow using `#[define_opaque]` on assoc methods and type aliases to override the default collection mode in
        // case it was capturing too much.
        self.mode = CollectionMode::RpitAndAsyncFnOnly;
    }
}

impl<'tcx> super::sig_types::SpannedTypeVisitor<'tcx> for OpaqueTypeCollector<'tcx> {
    #[instrument(skip(self), ret, level = "trace")]
    fn visit(&mut self, span: Span, value: impl TypeVisitable<TyCtxt<'tcx>>) {
        self.visit_spanned(span, value);
    }
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for OpaqueTypeCollector<'tcx> {
    #[instrument(skip(self), ret, level = "trace")]
    fn visit_ty(&mut self, t: Ty<'tcx>) {
        t.super_visit_with(self);
        match *t.kind() {
            ty::Alias(ty::Opaque, alias_ty) if alias_ty.def_id.is_local() => {
                self.visit_opaque_ty(alias_ty);
            }
            // Skips type aliases, as they are meant to be transparent.
            // FIXME(type_alias_impl_trait): can we require mentioning nested type aliases explicitly?
            ty::Alias(ty::Free, alias_ty) if let Some(def_id) = alias_ty.def_id.as_local() => {
                if !self.seen.insert(def_id) {
                    return;
                }
                self.tcx
                    .type_of(alias_ty.def_id)
                    .instantiate(self.tcx, alias_ty.args)
                    .visit_with(self);
            }
            ty::Alias(ty::Projection, alias_ty) => {
                // This avoids having to do normalization of `Self::AssocTy` by only
                // supporting the case of a method defining opaque types from assoc types
                // in the same impl block.
                if let Some(impl_trait_ref) = self.parent_impl_trait_ref() {
                    // If the trait ref of the associated item and the impl differs,
                    // then we can't use the impl's identity args below, so
                    // just skip.
                    if alias_ty.trait_ref(self.tcx) == impl_trait_ref {
                        let parent = self.parent().expect("we should have a parent here");

                        for &assoc in self.tcx.associated_items(parent).in_definition_order() {
                            trace!(?assoc);
                            if assoc.expect_trait_impl() != Ok(alias_ty.def_id) {
                                continue;
                            }

                            // If the type is further specializable, then the type_of
                            // is not actually correct below.
                            if !assoc.defaultness(self.tcx).is_final() {
                                continue;
                            }

                            if !self.seen.insert(assoc.def_id.expect_local()) {
                                return;
                            }

                            let alias_args = alias_ty.args.rebase_onto(
                                self.tcx,
                                impl_trait_ref.def_id,
                                ty::GenericArgs::identity_for_item(self.tcx, parent),
                            );

                            if self.tcx.check_args_compatible(assoc.def_id, alias_args) {
                                self.tcx
                                    .type_of(assoc.def_id)
                                    .instantiate(self.tcx, alias_args)
                                    .visit_with(self);
                                return;
                            } else {
                                self.tcx.dcx().span_delayed_bug(
                                    self.tcx.def_span(assoc.def_id),
                                    "item had incorrect args",
                                );
                            }
                        }
                    }
                } else if let Some(ty::ImplTraitInTraitData::Trait { fn_def_id, .. }) =
                    self.tcx.opt_rpitit_info(alias_ty.def_id)
                    && fn_def_id == self.item.into()
                {
                    // RPITIT in trait definitions get desugared to an associated type. For
                    // default methods we also create an opaque type this associated type
                    // normalizes to. The associated type is only known to normalize to the
                    // opaque if it is fully concrete. There could otherwise be an impl
                    // overwriting the default method.
                    //
                    // However, we have to be able to normalize the associated type while inside
                    // of the default method. This is normally handled by adding an unchecked
                    // `Projection(<Self as Trait>::synthetic_assoc_ty, trait_def::opaque)`
                    // assumption to the `param_env` of the default method. We also separately
                    // rely on that assumption here.
                    let ty = self.tcx.type_of(alias_ty.def_id).instantiate(self.tcx, alias_ty.args);
                    let ty::Alias(ty::Opaque, alias_ty) = *ty.kind() else { bug!("{ty:?}") };
                    self.visit_opaque_ty(alias_ty);
                }
            }
            _ => trace!(kind=?t.kind()),
        }
    }
}

fn opaque_types_defined_by<'tcx>(
    tcx: TyCtxt<'tcx>,
    item: LocalDefId,
) -> &'tcx ty::List<LocalDefId> {
    let kind = tcx.def_kind(item);
    trace!(?kind);
    let mut collector = OpaqueTypeCollector::new(tcx, item);
    collector.collect_taits_from_defines_attr();
    super::sig_types::walk_types(tcx, item, &mut collector);

    match kind {
        DefKind::AssocFn
        | DefKind::Fn
        | DefKind::Static { .. }
        | DefKind::Const
        | DefKind::AssocConst
        | DefKind::AnonConst => {
            collector.collect_taits_declared_in_body();
        }
        // Closures and coroutines are type checked with their parent
        // Note that we also support `SyntheticCoroutineBody` since we create
        // a MIR body for the def kind, and some MIR passes (like promotion)
        // may require doing analysis using its typing env.
        DefKind::Closure | DefKind::InlineConst | DefKind::SyntheticCoroutineBody => {
            collector.opaques.extend(tcx.opaque_types_defined_by(tcx.local_parent(item)));
        }
        DefKind::AssocTy | DefKind::TyAlias | DefKind::GlobalAsm => {}
        DefKind::OpaqueTy
        | DefKind::Mod
        | DefKind::Struct
        | DefKind::Union
        | DefKind::Enum
        | DefKind::Variant
        | DefKind::Trait
        | DefKind::ForeignTy
        | DefKind::TraitAlias
        | DefKind::TyParam
        | DefKind::ConstParam
        | DefKind::Ctor(_, _)
        | DefKind::Macro(_)
        | DefKind::ExternCrate
        | DefKind::Use
        | DefKind::ForeignMod
        | DefKind::Field
        | DefKind::LifetimeParam
        | DefKind::Impl { .. } => {
            span_bug!(
                tcx.def_span(item),
                "`opaque_types_defined_by` not defined for {} `{item:?}`",
                kind.descr(item.to_def_id())
            );
        }
    }
    tcx.mk_local_def_ids(&collector.opaques)
}

pub(super) fn provide(providers: &mut Providers) {
    *providers = Providers { opaque_types_defined_by, ..*providers };
}
