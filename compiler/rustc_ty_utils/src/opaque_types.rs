use rustc_data_structures::fx::FxHashSet;
use rustc_hir::intravisit::Visitor;
use rustc_hir::{def::DefKind, def_id::LocalDefId};
use rustc_hir::{intravisit, CRATE_HIR_ID};
use rustc_infer::infer::outlives::env::OutlivesEnvironment;
use rustc_infer::infer::{InferOk, TyCtxtInferExt};
use rustc_infer::traits::{TraitEngine, TraitEngineExt as _};
use rustc_middle::query::Providers;
use rustc_middle::traits::ObligationCause;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::ty::{TypeFoldable, TypeSuperVisitable, TypeVisitableExt, TypeVisitor};
use rustc_span::{ErrorGuaranteed, Span};
use rustc_trait_selection::traits::outlives_bounds::InferCtxtExt;
use rustc_trait_selection::traits::project::with_replaced_escaping_bound_vars;
use rustc_trait_selection::traits::{NormalizeExt, TraitEngineExt as _};
use std::ops::ControlFlow;

struct OpaqueTypeCollector<'tcx> {
    tcx: TyCtxt<'tcx>,

    opaques: Vec<LocalDefId>,

    /// The `DefId` of the item which we are collecting opaque types for.
    item: LocalDefId,

    /// Avoid infinite recursion due to recursive declarations.
    seen: FxHashSet<Ty<'tcx>>,

    span: Option<Span>,

    binder: ty::DebruijnIndex,
}

impl<'tcx> OpaqueTypeCollector<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, item: LocalDefId) -> Self {
        Self {
            tcx,
            opaques: Vec::new(),
            item,
            seen: Default::default(),
            span: None,
            binder: ty::INNERMOST,
        }
    }

    fn visit_spanned_after_normalizing(
        &mut self,
        span: Span,
        value: impl TypeFoldable<TyCtxt<'tcx>>,
    ) {
        let old = self.span;
        self.span = Some(span);

        if let Ok(value) = self.normalize_if_possible(span, value) {
            value.visit_with(self);
        }

        self.span = old;
    }

    fn normalize_if_possible<T: TypeFoldable<TyCtxt<'tcx>>>(
        &self,
        span: Span,
        value: T,
    ) -> Result<T, ErrorGuaranteed> {
        if !value.has_projections() {
            return Ok(value);
        }

        let infcx = self.tcx.infer_ctxt().build();
        let param_env = self.tcx.param_env(self.item);

        with_replaced_escaping_bound_vars(
            &infcx,
            &mut vec![None; self.binder.as_usize()],
            value,
            |value| {
                let mut fulfill_cx = <dyn TraitEngine<'tcx>>::new(&infcx);

                let normalized_value = match infcx
                    .at(&ObligationCause::misc(span, self.item), param_env)
                    .deeply_normalize(value, &mut *fulfill_cx)
                {
                    Ok(t) => t,
                    Err(errors) => {
                        return Err(self
                            .tcx
                            .sess
                            .delay_span_bug(span, format!("{errors:#?} in {:?}", self.item)));
                    }
                };

                let InferOk { value: implied_wf_types, obligations } =
                    infcx.at(&ObligationCause::misc(span, self.item), param_env).normalize(
                        self.tcx
                            .assumed_wf_types(self.item)
                            .iter()
                            .map(|(ty, _span)| *ty)
                            .collect::<Vec<_>>(),
                    );
                fulfill_cx.register_predicate_obligations(&infcx, obligations);

                let errors = fulfill_cx.select_all_or_error(&infcx);
                if !errors.is_empty() {
                    return Err(self.tcx.sess.delay_span_bug(span, format!("{errors:#?}")));
                }

                let outlives_env = OutlivesEnvironment::with_bounds(
                    param_env,
                    infcx.implied_bounds_tys(
                        param_env,
                        self.item,
                        implied_wf_types.into_iter().collect(),
                    ),
                );
                let errors = infcx.resolve_regions(&outlives_env);
                if !errors.is_empty() {
                    return Err(self.tcx.sess.delay_span_bug(span, format!("{errors:#?}")));
                }

                let resolved_value = match infcx.fully_resolve(normalized_value) {
                    Ok(resolved_value) => resolved_value,
                    Err(f) => {
                        return Err(self.tcx.sess.delay_span_bug(span, format!("{f:?}")));
                    }
                };

                Ok(resolved_value)
            },
        )
    }

    /// Returns `true` if `opaque_hir_id` is a sibling or a child of a sibling of `self.item`.
    ///
    /// Example:
    /// ```ignore UNSOLVED (is this a bug?)
    /// # #![feature(type_alias_impl_trait)]
    /// pub mod foo {
    ///     pub mod bar {
    ///         pub trait Bar { /* ... */ }
    ///         pub type Baz = impl Bar;
    ///
    ///         # impl Bar for () {}
    ///         fn f1() -> Baz { /* ... */ }
    ///     }
    ///     fn f2() -> bar::Baz { /* ... */ }
    /// }
    /// ```
    ///
    /// and `opaque_def_id` is the `DefId` of the definition of the opaque type `Baz`.
    /// For the above example, this function returns `true` for `f1` and `false` for `f2`.
    #[instrument(level = "trace", skip(self), ret)]
    fn check_tait_defining_scope(&self, opaque_def_id: LocalDefId) -> bool {
        let mut hir_id = self.tcx.hir().local_def_id_to_hir_id(self.item);
        let opaque_hir_id = self.tcx.hir().local_def_id_to_hir_id(opaque_def_id);

        // Named opaque types can be defined by any siblings or children of siblings.
        let scope = self.tcx.hir().get_defining_scope(opaque_hir_id);
        // We walk up the node tree until we hit the root or the scope of the opaque type.
        while hir_id != scope && hir_id != CRATE_HIR_ID {
            hir_id = self.tcx.hir().get_parent_item(hir_id).into();
        }
        // Syntactically, we are allowed to define the concrete type if:
        hir_id == scope
    }

    fn collect_body_and_predicate_taits(&mut self) {
        // Look at all where bounds.
        for (pred, span) in self.tcx.predicates_of(self.item).instantiate_identity(self.tcx) {
            self.visit_spanned_after_normalizing(span, pred);
        }
        // An item is allowed to constrain opaques declared within its own body (but not nested within
        // nested functions).
        self.collect_taits_declared_in_body();
    }

    #[instrument(level = "trace", skip(self))]
    fn collect_taits_declared_in_body(&mut self) {
        let body = self.tcx.hir().body(self.tcx.hir().body_owned_by(self.item)).value;
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
                let body = self.collector.tcx.hir().body(id);
                self.visit_body(body);
            }
        }
        TaitInBodyFinder { collector: self }.visit_expr(body);
    }
}

impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for OpaqueTypeCollector<'tcx> {
    fn visit_binder<T: ty::TypeVisitable<TyCtxt<'tcx>>>(
        &mut self,
        t: &ty::Binder<'tcx, T>,
    ) -> ControlFlow<!> {
        self.binder.shift_in(1);
        t.super_visit_with(self);
        self.binder.shift_out(1);
        ControlFlow::Continue(())
    }

    #[instrument(skip(self), ret, level = "trace")]
    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<!> {
        // Erase all free and escaping bound regions, to make sure that
        // we're not walking into a type that contains itself modulo
        // regions.
        if !self.seen.insert(self.tcx.fold_regions(t, |_, _| self.tcx.lifetimes.re_erased)) {
            return ControlFlow::Continue(());
        }

        t.super_visit_with(self)?;
        match t.kind() {
            ty::Alias(ty::Opaque, alias_ty) if alias_ty.def_id.is_local() => {
                // TAITs outside their defining scopes are ignored.
                let origin = self.tcx.opaque_type_origin(alias_ty.def_id.expect_local());
                trace!(?origin);
                match origin {
                    rustc_hir::OpaqueTyOrigin::FnReturn(_)
                    | rustc_hir::OpaqueTyOrigin::AsyncFn(_) => {}
                    rustc_hir::OpaqueTyOrigin::TyAlias { in_assoc_ty } => {
                        if in_assoc_ty {
                            // Make sure that the TAIT comes from an associated item
                            // in the same implementation.
                            let Some(assoc_item) =
                                self.tcx.opt_associated_item(self.item.to_def_id())
                            else {
                                return ControlFlow::Continue(());
                            };
                            let mut tait_parent = self.tcx.parent(alias_ty.def_id);
                            while self.tcx.def_kind(tait_parent) == DefKind::OpaqueTy {
                                tait_parent = self.tcx.parent(tait_parent);
                            }
                            if self.tcx.parent(tait_parent) != assoc_item.container_id(self.tcx) {
                                return ControlFlow::Continue(());
                            }
                        } else {
                            // Otherwise, the TAIT must be a sibling of the item.
                            if !self.check_tait_defining_scope(alias_ty.def_id.expect_local()) {
                                return ControlFlow::Continue(());
                            }
                        }
                    }
                }

                self.opaques.push(alias_ty.def_id.expect_local());

                // Collect opaque types nested within the associated type bounds of this opaque type.
                for (pred, span) in self
                    .tcx
                    .explicit_item_bounds(alias_ty.def_id)
                    .iter_instantiated_copied(self.tcx, alias_ty.args)
                {
                    trace!(?pred);
                    self.visit_spanned_after_normalizing(span, pred);
                }
            }
            ty::Adt(def, args) if def.did().is_local() => {
                for variant in def.variants().iter() {
                    for field in variant.fields.iter() {
                        let ty = self.tcx.type_of(field.did).instantiate(self.tcx, args);
                        self.visit_spanned_after_normalizing(self.tcx.def_span(field.did), ty);
                    }
                }
            }
            _ => trace!(kind=?t.kind()),
        }

        ControlFlow::Continue(())
    }
}

fn opaque_types_defined_by<'tcx>(tcx: TyCtxt<'tcx>, item: LocalDefId) -> &'tcx [LocalDefId] {
    let kind = tcx.def_kind(item);
    trace!(?kind);
    let mut collector = OpaqueTypeCollector::new(tcx, item);
    match kind {
        // Walk over the signature of the function-like to find the opaques.
        DefKind::AssocFn | DefKind::Fn => {
            let ty_sig = tcx.liberate_late_bound_regions(
                item.to_def_id(),
                tcx.fn_sig(item).instantiate_identity(),
            );
            let hir_sig = tcx.hir().get_by_def_id(item).fn_sig().unwrap();
            // Walk over the inputs and outputs manually in order to get good spans for them.
            collector.visit_spanned_after_normalizing(hir_sig.decl.output.span(), ty_sig.output());
            for (hir, ty) in hir_sig.decl.inputs.iter().zip(ty_sig.inputs().iter()) {
                collector.visit_spanned_after_normalizing(hir.span, *ty);
            }
            collector.collect_body_and_predicate_taits();
        }
        // Walk over the type of the item to find opaques.
        DefKind::Static(_) | DefKind::Const | DefKind::AssocConst | DefKind::AnonConst => {
            let span = match tcx.hir().get_by_def_id(item).ty() {
                Some(ty) => ty.span,
                _ => tcx.def_span(item),
            };
            collector
                .visit_spanned_after_normalizing(span, tcx.type_of(item).instantiate_identity());
            collector.collect_body_and_predicate_taits();
        }
        // We're also doing this for `AssocTy` for the wf checks in `check_opaque_meets_bounds`
        DefKind::TyAlias | DefKind::AssocTy => {
            let span = match tcx.hir().get_by_def_id(item).ty() {
                Some(ty) => ty.span,
                _ => tcx.def_span(item),
            };
            collector
                .visit_spanned_after_normalizing(span, tcx.type_of(item).instantiate_identity());
        }
        DefKind::OpaqueTy => {
            for (pred, span) in tcx.explicit_item_bounds(item).instantiate_identity_iter_copied() {
                collector.visit_spanned_after_normalizing(span, pred);
            }
        }
        DefKind::Mod
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
        | DefKind::GlobalAsm
        | DefKind::Impl { .. } => {}
        // Closures and generators are type checked with their parent, so there is no difference here.
        DefKind::Closure | DefKind::Generator | DefKind::InlineConst => {
            return tcx.opaque_types_defined_by(tcx.local_parent(item));
        }
    }
    tcx.arena.alloc_from_iter(collector.opaques)
}

pub(super) fn provide(providers: &mut Providers) {
    *providers = Providers { opaque_types_defined_by, ..*providers };
}
