use rustc_ast::visit::{self, BoundKind, LifetimeCtxt, Visitor};
use rustc_ast::{
    GenericBound, GenericBounds, Lifetime, NodeId, PathSegment, PolyTraitRef, Ty, TyKind,
};
use rustc_data_structures::fx::FxIndexSet;
use rustc_hir::def::{DefKind, LifetimeRes, Res};
use rustc_middle::span_bug;
use rustc_middle::ty::ResolverAstLowering;
use rustc_span::Span;
use rustc_span::symbol::{Ident, kw};

use super::ResolverAstLoweringExt;

struct LifetimeCollectVisitor<'ast> {
    resolver: &'ast mut ResolverAstLowering,
    always_capture_in_scope: bool,
    current_binders: Vec<NodeId>,
    collected_lifetimes: FxIndexSet<Lifetime>,
}

impl<'ast> LifetimeCollectVisitor<'ast> {
    fn new(resolver: &'ast mut ResolverAstLowering, always_capture_in_scope: bool) -> Self {
        Self {
            resolver,
            always_capture_in_scope,
            current_binders: Vec::new(),
            collected_lifetimes: FxIndexSet::default(),
        }
    }

    fn visit_opaque(&mut self, opaque_ty_node_id: NodeId, bounds: &'ast GenericBounds, span: Span) {
        // If we're edition 2024 or within a TAIT or RPITIT, *and* there is no
        // `use<>` statement to override the default capture behavior, then
        // capture all of the in-scope lifetimes.
        if (self.always_capture_in_scope || span.at_least_rust_2024())
            && bounds.iter().all(|bound| !matches!(bound, GenericBound::Use(..)))
        {
            for (ident, id, _) in self.resolver.extra_lifetime_params(opaque_ty_node_id) {
                self.record_lifetime_use(Lifetime { id, ident });
            }
        }

        // We also recurse on the bounds to make sure we capture all the lifetimes
        // mentioned in the bounds. These may disagree with the `use<>` list, in which
        // case we will error on these later. We will also recurse to visit any
        // nested opaques, which may *implicitly* capture lifetimes.
        for bound in bounds {
            self.visit_param_bound(bound, BoundKind::Bound);
        }
    }

    fn record_lifetime_use(&mut self, lifetime: Lifetime) {
        match self.resolver.get_lifetime_res(lifetime.id).unwrap_or(LifetimeRes::Error) {
            LifetimeRes::Param { binder, .. } | LifetimeRes::Fresh { binder, .. } => {
                if !self.current_binders.contains(&binder) {
                    self.collected_lifetimes.insert(lifetime);
                }
            }
            LifetimeRes::Static { .. } | LifetimeRes::Error => {
                self.collected_lifetimes.insert(lifetime);
            }
            LifetimeRes::Infer => {}
            res => {
                let bug_msg = format!(
                    "Unexpected lifetime resolution {:?} for {:?} at {:?}",
                    res, lifetime.ident, lifetime.ident.span
                );
                span_bug!(lifetime.ident.span, "{}", bug_msg);
            }
        }
    }

    /// This collect lifetimes that are elided, for nodes like `Foo<T>` where there are no explicit
    /// lifetime nodes. Is equivalent to having "pseudo" nodes introduced for each of the node ids
    /// in the list start..end.
    fn record_elided_anchor(&mut self, node_id: NodeId, span: Span) {
        if let Some(LifetimeRes::ElidedAnchor { start, end }) =
            self.resolver.get_lifetime_res(node_id)
        {
            for i in start..end {
                let lifetime = Lifetime { id: i, ident: Ident::new(kw::UnderscoreLifetime, span) };
                self.record_lifetime_use(lifetime);
            }
        }
    }
}

impl<'ast> Visitor<'ast> for LifetimeCollectVisitor<'ast> {
    fn visit_lifetime(&mut self, lifetime: &'ast Lifetime, _: LifetimeCtxt) {
        self.record_lifetime_use(*lifetime);
    }

    fn visit_path_segment(&mut self, path_segment: &'ast PathSegment) {
        self.record_elided_anchor(path_segment.id, path_segment.ident.span);
        visit::walk_path_segment(self, path_segment);
    }

    fn visit_poly_trait_ref(&mut self, t: &'ast PolyTraitRef) {
        self.current_binders.push(t.trait_ref.ref_id);

        visit::walk_poly_trait_ref(self, t);

        self.current_binders.pop();
    }

    fn visit_ty(&mut self, t: &'ast Ty) {
        match &t.kind {
            TyKind::Path(None, _) => {
                // We can sometimes encounter bare trait objects
                // which are represented in AST as paths.
                if let Some(partial_res) = self.resolver.get_partial_res(t.id)
                    && let Some(Res::Def(DefKind::Trait | DefKind::TraitAlias, _)) =
                        partial_res.full_res()
                {
                    self.current_binders.push(t.id);
                    visit::walk_ty(self, t);
                    self.current_binders.pop();
                } else {
                    visit::walk_ty(self, t);
                }
            }
            TyKind::BareFn(_) => {
                self.current_binders.push(t.id);
                visit::walk_ty(self, t);
                self.current_binders.pop();
            }
            TyKind::Ref(None, _) | TyKind::PinnedRef(None, _) => {
                self.record_elided_anchor(t.id, t.span);
                visit::walk_ty(self, t);
            }
            TyKind::ImplTrait(opaque_ty_node_id, bounds) => {
                self.visit_opaque(*opaque_ty_node_id, bounds, t.span)
            }
            _ => {
                visit::walk_ty(self, t);
            }
        }
    }
}

pub(crate) fn lifetimes_for_opaque(
    resolver: &mut ResolverAstLowering,
    always_capture_in_scope: bool,
    opaque_ty_node_id: NodeId,
    bounds: &GenericBounds,
    span: Span,
) -> FxIndexSet<Lifetime> {
    let mut visitor = LifetimeCollectVisitor::new(resolver, always_capture_in_scope);
    visitor.visit_opaque(opaque_ty_node_id, bounds, span);
    visitor.collected_lifetimes
}
