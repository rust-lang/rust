use super::ResolverAstLoweringExt;
use rustc_ast::visit::{self, BoundKind, LifetimeCtxt, Visitor};
use rustc_ast::{
    FnRetTy, GenericBounds, Lifetime, NodeId, PathSegment, PolyTraitRef, TraitBoundModifier, Ty,
    TyKind,
};
use rustc_hir::def::LifetimeRes;
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
        let res = self.resolver.get_lifetime_res(lifetime.id).unwrap_or(LifetimeRes::Error);

        if res.binder().map_or(true, |b| !self.current_binders.contains(&b)) {
            if !self.collected_lifetimes.contains(&lifetime) {
                self.collected_lifetimes.push(lifetime);
            }
        }
    }
}

impl<'ast> Visitor<'ast> for LifetimeCollectVisitor<'ast> {
    fn visit_lifetime(&mut self, lifetime: &'ast Lifetime, _: LifetimeCtxt) {
        self.record_lifetime_use(*lifetime);
    }

    fn visit_path_segment(&mut self, path_span: Span, path_segment: &'ast PathSegment) {
        if let Some(LifetimeRes::ElidedAnchor { start, end }) =
            self.resolver.get_lifetime_res(path_segment.id)
        {
            for i in start..end {
                let lifetime =
                    Lifetime { id: i, ident: Ident::new(kw::UnderscoreLifetime, path_span) };
                self.record_lifetime_use(lifetime);
            }
        }
        visit::walk_path_segment(self, path_span, path_segment);
    }

    fn visit_poly_trait_ref(&mut self, t: &'ast PolyTraitRef, m: &'ast TraitBoundModifier) {
        self.current_binders.push(t.trait_ref.ref_id);

        visit::walk_poly_trait_ref(self, t, m);

        self.current_binders.pop();
    }

    fn visit_ty(&mut self, t: &'ast Ty) {
        match t.kind {
            TyKind::BareFn(_) => {
                self.current_binders.push(t.id);
                visit::walk_ty(self, t);
                self.current_binders.pop();
            }
            _ => {
                visit::walk_ty(self, t);
            }
        }
    }
}

pub fn lifetimes_in_ret_ty(resolver: &ResolverAstLowering, ret_ty: &FnRetTy) -> Vec<Lifetime> {
    let mut visitor = LifetimeCollectVisitor::new(resolver);
    visitor.visit_fn_ret_ty(ret_ty);
    visitor.collected_lifetimes
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
