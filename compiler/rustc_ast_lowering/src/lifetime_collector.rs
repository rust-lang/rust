use super::ResolverAstLoweringExt;
use rustc_ast::visit::{self, BoundKind, LifetimeCtxt, Visitor};
use rustc_ast::{GenericBounds, Lifetime, NodeId, PathSegment, PolyTraitRef, Ty, TyKind};
use rustc_hir::def::{DefKind, LifetimeRes, Res};
use rustc_middle::span_bug;
use rustc_middle::ty::ResolverAstLowering;
use rustc_span::symbol::{kw, Ident};
use rustc_span::Span;

struct LifetimeCollectVisitor<'ast> {
    resolver: &'ast ResolverAstLowering,
    current_binders: Vec<NodeId>,
    collected_lifetimes: Vec<Lifetime>,
}

impl<'ast> LifetimeCollectVisitor<'ast> {
    fn new(resolver: &'ast ResolverAstLowering) -> Self {
        Self { resolver, current_binders: Vec::new(), collected_lifetimes: Vec::new() }
    }

    fn record_lifetime_use(&mut self, lifetime: Lifetime) {
        match self.resolver.get_lifetime_res(lifetime.id).unwrap_or(LifetimeRes::Error) {
            LifetimeRes::Param { binder, .. } | LifetimeRes::Fresh { binder, .. } => {
                if !self.current_binders.contains(&binder) {
                    if !self.collected_lifetimes.contains(&lifetime) {
                        self.collected_lifetimes.push(lifetime);
                    }
                }
            }
            LifetimeRes::Static | LifetimeRes::Error => {
                if !self.collected_lifetimes.contains(&lifetime) {
                    self.collected_lifetimes.push(lifetime);
                }
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
            TyKind::Ref(None, _) => {
                self.record_elided_anchor(t.id, t.span);
                visit::walk_ty(self, t);
            }
            _ => {
                visit::walk_ty(self, t);
            }
        }
    }
}

pub fn lifetimes_in_bounds(
    resolver: &ResolverAstLowering,
    bounds: &GenericBounds,
) -> Vec<Lifetime> {
    let mut visitor = LifetimeCollectVisitor::new(resolver);
    for bound in bounds {
        visitor.visit_param_bound(bound, BoundKind::Bound);
    }
    visitor.collected_lifetimes
}
