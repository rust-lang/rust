// Simply gives a rough count of the number of nodes in an AST.

use rustc_ast::visit::*;
use rustc_ast::*;
use rustc_span::symbol::Ident;
use rustc_span::Span;

pub struct NodeCounter {
    pub count: usize,
}

impl NodeCounter {
    pub fn new() -> NodeCounter {
        NodeCounter { count: 0 }
    }
}

impl<'ast> Visitor<'ast> for NodeCounter {
    fn visit_ident(&mut self, _ident: Ident) {
        self.count += 1;
    }
    fn visit_foreign_item(&mut self, i: &ForeignItem) {
        self.count += 1;
        walk_foreign_item(self, i)
    }
    fn visit_item(&mut self, i: &Item) {
        self.count += 1;
        walk_item(self, i)
    }
    fn visit_local(&mut self, l: &Local) {
        self.count += 1;
        walk_local(self, l)
    }
    fn visit_block(&mut self, b: &Block) {
        self.count += 1;
        walk_block(self, b)
    }
    fn visit_stmt(&mut self, s: &Stmt) {
        self.count += 1;
        walk_stmt(self, s)
    }
    fn visit_arm(&mut self, a: &Arm) {
        self.count += 1;
        walk_arm(self, a)
    }
    fn visit_pat(&mut self, p: &Pat) {
        self.count += 1;
        walk_pat(self, p)
    }
    fn visit_expr(&mut self, ex: &Expr) {
        self.count += 1;
        walk_expr(self, ex)
    }
    fn visit_ty(&mut self, t: &Ty) {
        self.count += 1;
        walk_ty(self, t)
    }
    fn visit_generic_param(&mut self, param: &GenericParam) {
        self.count += 1;
        walk_generic_param(self, param)
    }
    fn visit_generics(&mut self, g: &Generics) {
        self.count += 1;
        walk_generics(self, g)
    }
    fn visit_fn(&mut self, fk: visit::FnKind<'_>, _: Span, _: NodeId) {
        self.count += 1;
        walk_fn(self, fk)
    }
    fn visit_assoc_item(&mut self, ti: &AssocItem, ctxt: AssocCtxt) {
        self.count += 1;
        walk_assoc_item(self, ti, ctxt);
    }
    fn visit_trait_ref(&mut self, t: &TraitRef) {
        self.count += 1;
        walk_trait_ref(self, t)
    }
    fn visit_param_bound(&mut self, bounds: &GenericBound, _ctxt: BoundKind) {
        self.count += 1;
        walk_param_bound(self, bounds)
    }
    fn visit_poly_trait_ref(&mut self, t: &PolyTraitRef) {
        self.count += 1;
        walk_poly_trait_ref(self, t)
    }
    fn visit_variant_data(&mut self, s: &VariantData) {
        self.count += 1;
        walk_struct_def(self, s)
    }
    fn visit_field_def(&mut self, s: &FieldDef) {
        self.count += 1;
        walk_field_def(self, s)
    }
    fn visit_enum_def(&mut self, enum_definition: &EnumDef) {
        self.count += 1;
        walk_enum_def(self, enum_definition)
    }
    fn visit_variant(&mut self, v: &Variant) {
        self.count += 1;
        walk_variant(self, v)
    }
    fn visit_lifetime(&mut self, lifetime: &Lifetime, _: visit::LifetimeCtxt) {
        self.count += 1;
        walk_lifetime(self, lifetime)
    }
    fn visit_mac_call(&mut self, mac: &MacCall) {
        self.count += 1;
        walk_mac(self, mac)
    }
    fn visit_path(&mut self, path: &Path, _id: NodeId) {
        self.count += 1;
        walk_path(self, path)
    }
    fn visit_use_tree(&mut self, use_tree: &UseTree, id: NodeId, _nested: bool) {
        self.count += 1;
        walk_use_tree(self, use_tree, id)
    }
    fn visit_generic_args(&mut self, generic_args: &GenericArgs) {
        self.count += 1;
        walk_generic_args(self, generic_args)
    }
    fn visit_assoc_constraint(&mut self, constraint: &AssocConstraint) {
        self.count += 1;
        walk_assoc_constraint(self, constraint)
    }
    fn visit_attribute(&mut self, _attr: &Attribute) {
        self.count += 1;
    }
}
