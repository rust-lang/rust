use rustc_ast::visit::{self, BoundKind, LifetimeCtxt, Visitor};
use rustc_ast::{
    FnRetTy, GenericBounds, Lifetime, NodeId, PolyTraitRef, TraitBoundModifier, Ty, TyKind,
};
use rustc_data_structures::fx::FxHashMap;

struct LifetimeCollectVisitor<'ast> {
    current_binders: Vec<NodeId>,
    binders_to_ignore: FxHashMap<NodeId, Vec<NodeId>>,
    collected_lifetimes: Vec<&'ast Lifetime>,
}

impl<'ast> Visitor<'ast> for LifetimeCollectVisitor<'ast> {
    fn visit_lifetime(&mut self, lifetime: &'ast Lifetime, _: LifetimeCtxt) {
        if !self.collected_lifetimes.contains(&lifetime) {
            self.collected_lifetimes.push(lifetime);
        }
        self.binders_to_ignore.insert(lifetime.id, self.current_binders.clone());
    }

    fn visit_poly_trait_ref(&mut self, t: &'ast PolyTraitRef, m: &'ast TraitBoundModifier) {
        self.current_binders.push(t.trait_ref.ref_id);

        visit::walk_poly_trait_ref(self, t, m);

        self.current_binders.pop();
    }

    fn visit_ty(&mut self, t: &'ast Ty) {
        if let TyKind::BareFn(_) = t.kind {
            self.current_binders.push(t.id);
        }
        visit::walk_ty(self, t);
        if let TyKind::BareFn(_) = t.kind {
            self.current_binders.pop();
        }
    }
}

pub fn lifetimes_in_ret_ty(ret_ty: &FnRetTy) -> (Vec<&Lifetime>, FxHashMap<NodeId, Vec<NodeId>>) {
    let mut visitor = LifetimeCollectVisitor {
        current_binders: Vec::new(),
        binders_to_ignore: FxHashMap::default(),
        collected_lifetimes: Vec::new(),
    };
    visitor.visit_fn_ret_ty(ret_ty);
    (visitor.collected_lifetimes, visitor.binders_to_ignore)
}

pub fn lifetimes_in_bounds(
    bounds: &GenericBounds,
) -> (Vec<&Lifetime>, FxHashMap<NodeId, Vec<NodeId>>) {
    let mut visitor = LifetimeCollectVisitor {
        current_binders: Vec::new(),
        binders_to_ignore: FxHashMap::default(),
        collected_lifetimes: Vec::new(),
    };
    for bound in bounds {
        visitor.visit_param_bound(bound, BoundKind::Bound);
    }
    (visitor.collected_lifetimes, visitor.binders_to_ignore)
}
