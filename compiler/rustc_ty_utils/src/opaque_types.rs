use rustc_data_structures::fx::FxHashSet;
use rustc_hir::intravisit::Visitor;
use rustc_hir::{def::DefKind, def_id::LocalDefId};
use rustc_hir::{intravisit, CRATE_HIR_ID};
use rustc_middle::query::Providers;
use rustc_middle::ty::util::{CheckRegions, NotUniqueParam};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_middle::ty::{TypeSuperVisitable, TypeVisitable, TypeVisitor};
use rustc_span::Span;
use rustc_trait_selection::traits::check_substs_compatible;
use std::ops::ControlFlow;

use crate::errors::{DuplicateArg, NotParam};

struct OpaqueTypeCollector<'tcx> {
    tcx: TyCtxt<'tcx>,
    opaques: Vec<LocalDefId>,
    /// The `DefId` of the item which we are collecting opaque types for.
    item: LocalDefId,

    /// Avoid infinite recursion due to recursive declarations.
    seen: FxHashSet<LocalDefId>,

    span: Option<Span>,
}

impl<'tcx> OpaqueTypeCollector<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, item: LocalDefId) -> Self {
        Self { tcx, opaques: Vec::new(), item, seen: Default::default(), span: None }
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

    fn parent_trait_ref(&self) -> Option<ty::TraitRef<'tcx>> {
        let parent = self.parent()?;
        if matches!(self.tcx.def_kind(parent), DefKind::Impl { .. }) {
            Some(self.tcx.impl_trait_ref(parent)?.subst_identity())
        } else {
            None
        }
    }

    fn parent(&self) -> Option<LocalDefId> {
        match self.tcx.def_kind(self.item) {
            DefKind::AnonConst | DefKind::InlineConst | DefKind::Fn | DefKind::TyAlias => None,
            DefKind::AssocFn | DefKind::AssocTy | DefKind::AssocConst => {
                Some(self.tcx.local_parent(self.item))
            }
            other => span_bug!(
                self.tcx.def_span(self.item),
                "unhandled item with opaque types: {other:?}"
            ),
        }
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
        self.tcx.predicates_of(self.item).instantiate_identity(self.tcx).visit_with(self);
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
    #[instrument(skip(self), ret, level = "trace")]
    fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<!> {
        t.super_visit_with(self)?;
        match t.kind() {
            ty::Alias(ty::Opaque, alias_ty) if alias_ty.def_id.is_local() => {
                if !self.seen.insert(alias_ty.def_id.expect_local()) {
                    return ControlFlow::Continue(());
                }

                // TAITs outside their defining scopes are ignored.
                let origin = self.tcx.opaque_type_origin(alias_ty.def_id.expect_local());
                trace!(?origin);
                match origin {
                    rustc_hir::OpaqueTyOrigin::FnReturn(_)
                    | rustc_hir::OpaqueTyOrigin::AsyncFn(_) => {}
                    rustc_hir::OpaqueTyOrigin::TyAlias { in_assoc_ty } => {
                        if !in_assoc_ty {
                            if !self.check_tait_defining_scope(alias_ty.def_id.expect_local()) {
                                return ControlFlow::Continue(());
                            }
                        }
                    }
                }

                self.opaques.push(alias_ty.def_id.expect_local());

                match self.tcx.uses_unique_generic_params(alias_ty.substs, CheckRegions::Bound) {
                    Ok(()) => {
                        // FIXME: implement higher kinded lifetime bounds on nested opaque types. They are not
                        // supported at all, so this is sound to do, but once we want to support them, you'll
                        // start seeing the error below.

                        // Collect opaque types nested within the associated type bounds of this opaque type.
                        // We use identity substs here, because we already know that the opaque type uses
                        // only generic parameters, and thus substituting would not give us more information.
                        for (pred, span) in self
                            .tcx
                            .explicit_item_bounds(alias_ty.def_id)
                            .subst_identity_iter_copied()
                        {
                            trace!(?pred);
                            self.visit_spanned(span, pred);
                        }
                    }
                    Err(NotUniqueParam::NotParam(arg)) => {
                        self.tcx.sess.emit_err(NotParam {
                            arg,
                            span: self.span(),
                            opaque_span: self.tcx.def_span(alias_ty.def_id),
                        });
                    }
                    Err(NotUniqueParam::DuplicateParam(arg)) => {
                        self.tcx.sess.emit_err(DuplicateArg {
                            arg,
                            span: self.span(),
                            opaque_span: self.tcx.def_span(alias_ty.def_id),
                        });
                    }
                }
            }
            ty::Alias(ty::Weak, alias_ty) if alias_ty.def_id.is_local() => {
                self.tcx
                    .type_of(alias_ty.def_id)
                    .subst(self.tcx, alias_ty.substs)
                    .visit_with(self)?;
            }
            ty::Alias(ty::Projection, alias_ty) => {
                // This avoids having to do normalization of `Self::AssocTy` by only
                // supporting the case of a method defining opaque types from assoc types
                // in the same impl block.
                if let Some(parent_trait_ref) = self.parent_trait_ref() {
                    // If the trait ref of the associated item and the impl differs,
                    // then we can't use the impl's identity substitutions below, so
                    // just skip.
                    if alias_ty.trait_ref(self.tcx) == parent_trait_ref {
                        let parent = self.parent().expect("we should have a parent here");

                        for &assoc in self.tcx.associated_items(parent).in_definition_order() {
                            trace!(?assoc);
                            if assoc.trait_item_def_id != Some(alias_ty.def_id) {
                                continue;
                            }

                            // If the type is further specializable, then the type_of
                            // is not actually correct below.
                            if !assoc.defaultness(self.tcx).is_final() {
                                continue;
                            }

                            let impl_substs = alias_ty.substs.rebase_onto(
                                self.tcx,
                                parent_trait_ref.def_id,
                                ty::InternalSubsts::identity_for_item(self.tcx, parent),
                            );

                            if check_substs_compatible(self.tcx, assoc, impl_substs) {
                                return self
                                    .tcx
                                    .type_of(assoc.def_id)
                                    .subst(self.tcx, impl_substs)
                                    .visit_with(self);
                            } else {
                                self.tcx.sess.delay_span_bug(
                                    self.tcx.def_span(assoc.def_id),
                                    "item had incorrect substs",
                                );
                            }
                        }
                    }
                }
            }
            ty::Adt(def, _) if def.did().is_local() => {
                if !self.seen.insert(def.did().expect_local()) {
                    return ControlFlow::Continue(());
                }
                for variant in def.variants().iter() {
                    for field in variant.fields.iter() {
                        // Don't use the `ty::Adt` substs, we either
                        // * found the opaque in the substs
                        // * will find the opaque in the unsubstituted fields
                        // The only other situation that can occur is that after substituting,
                        // some projection resolves to an opaque that we would have otherwise
                        // not found. While we could substitute and walk those, that would mean we
                        // would have to walk all substitutions of an Adt, which can quickly
                        // degenerate into looking at an exponential number of types.
                        let ty = self.tcx.type_of(field.did).subst_identity();
                        self.visit_spanned(self.tcx.def_span(field.did), ty);
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
            let ty_sig = tcx.fn_sig(item).subst_identity();
            let hir_sig = tcx.hir().get_by_def_id(item).fn_sig().unwrap();
            // Walk over the inputs and outputs manually in order to get good spans for them.
            collector.visit_spanned(hir_sig.decl.output.span(), ty_sig.output());
            for (hir, ty) in hir_sig.decl.inputs.iter().zip(ty_sig.inputs().iter()) {
                collector.visit_spanned(hir.span, ty.map_bound(|x| *x));
            }
            collector.collect_body_and_predicate_taits();
        }
        // Walk over the type of the item to find opaques.
        DefKind::Static(_) | DefKind::Const | DefKind::AssocConst | DefKind::AnonConst => {
            let span = match tcx.hir().get_by_def_id(item).ty() {
                Some(ty) => ty.span,
                _ => tcx.def_span(item),
            };
            collector.visit_spanned(span, tcx.type_of(item).subst_identity());
            collector.collect_body_and_predicate_taits();
        }
        // We're also doing this for `AssocTy` for the wf checks in `check_opaque_meets_bounds`
        DefKind::TyAlias | DefKind::AssocTy => {
            tcx.type_of(item).subst_identity().visit_with(&mut collector);
        }
        DefKind::OpaqueTy => {
            for (pred, span) in tcx.explicit_item_bounds(item).subst_identity_iter_copied() {
                collector.visit_spanned(span, pred);
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
        | DefKind::ImplTraitPlaceholder
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
