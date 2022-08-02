use super::ResolverAstLoweringExt;
use rustc_ast::visit::{self, BoundKind, LifetimeCtxt, Visitor};
use rustc_ast::{
    FnRetTy, GenericBounds, Lifetime, NodeId, PolyTraitRef, TraitBoundModifier, Ty, TyKind,
};
use rustc_hir::def::LifetimeRes;
use rustc_middle::ty::ResolverAstLowering;

struct LifetimeCollectVisitor<'this, 'ast: 'this> {
    resolver: &'this ResolverAstLowering,
    current_binders: Vec<NodeId>,
    collected_lifetimes: Vec<&'ast Lifetime>,
}

impl<'this, 'ast: 'this> LifetimeCollectVisitor<'this, 'ast> {
    fn new(resolver: &'this ResolverAstLowering) -> Self {
        Self { resolver, current_binders: Vec::new(), collected_lifetimes: Vec::new() }
    }
}

impl<'this, 'ast: 'this> Visitor<'ast> for LifetimeCollectVisitor<'this, 'ast> {
    fn visit_lifetime(&mut self, lifetime: &'ast Lifetime, _: LifetimeCtxt) {
        let res = self.resolver.get_lifetime_res(lifetime.id).unwrap_or(LifetimeRes::Error);

        if res.binder().map_or(true, |b| !self.current_binders.contains(&b)) {
            if !self.collected_lifetimes.contains(&lifetime) {
                self.collected_lifetimes.push(lifetime);
            }
        }
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

pub fn lifetimes_in_ret_ty<'this, 'ast: 'this>(
    resolver: &'this ResolverAstLowering,
    ret_ty: &'ast FnRetTy,
) -> Vec<&'ast Lifetime> {
    let mut visitor = LifetimeCollectVisitor::new(resolver);
    visitor.visit_fn_ret_ty(ret_ty);
    visitor.collected_lifetimes
}

pub fn lifetimes_in_bounds<'this, 'ast: 'this>(
    resolver: &'this ResolverAstLowering,
    bounds: &'ast GenericBounds,
) -> Vec<&'ast Lifetime> {
    let mut visitor = LifetimeCollectVisitor::new(resolver);
    for bound in bounds {
        visitor.visit_param_bound(bound, BoundKind::Bound);
    }
    visitor.collected_lifetimes
}
