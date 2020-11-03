//! AST walker. Each overridden visit method has full control over what
//! happens with its node, it can do its own traversal of the node's children,
//! call `visit::walk_*` to apply the default traversal algorithm, or prevent
//! deeper traversal by doing nothing.
//!
//! Note: it is an important invariant that the default visitor walks the body
//! of a function in "execution order" (more concretely, reverse post-order
//! with respect to the CFG implied by the AST), meaning that if AST node A may
//! execute before AST node B, then A is visited first. The borrow checker in
//! particular relies on this property.
//!
//! Note: walking an AST before macro expansion is probably a bad idea. For
//! instance, a walker looking for item names in a module will miss all of
//! those that are created by the expansion of a macro.

use crate::ast::*;
use crate::token;
use crate::tokenstream::TokenTree;

use rustc_span::symbol::{Ident, Symbol};
use rustc_span::Span;

#[derive(Copy, Clone, PartialEq)]
pub enum AssocCtxt {
    Trait,
    Impl,
}

#[derive(Copy, Clone, PartialEq)]
pub enum FnCtxt {
    Free,
    Foreign,
    Assoc(AssocCtxt),
}

#[derive(Copy, Clone)]
pub enum FnKind<'a> {
    /// E.g., `fn foo()`, `fn foo(&self)`, or `extern "Abi" fn foo()`.
    Fn(FnCtxt, Ident, &'a FnSig, &'a Visibility, Option<&'a Block>),

    /// E.g., `|x, y| body`.
    Closure(&'a FnDecl, &'a Expr),
}

impl<'a> FnKind<'a> {
    pub fn header(&self) -> Option<&'a FnHeader> {
        match *self {
            FnKind::Fn(_, _, sig, _, _) => Some(&sig.header),
            FnKind::Closure(_, _) => None,
        }
    }

    pub fn ident(&self) -> Option<&Ident> {
        match self {
            FnKind::Fn(_, ident, ..) => Some(ident),
            _ => None,
        }
    }

    pub fn decl(&self) -> &'a FnDecl {
        match self {
            FnKind::Fn(_, _, sig, _, _) => &sig.decl,
            FnKind::Closure(decl, _) => decl,
        }
    }

    pub fn ctxt(&self) -> Option<FnCtxt> {
        match self {
            FnKind::Fn(ctxt, ..) => Some(*ctxt),
            FnKind::Closure(..) => None,
        }
    }
}

/// Each method of the `Visitor` trait is a hook to be potentially
/// overridden. Each method's default implementation recursively visits
/// the substructure of the input via the corresponding `walk` method;
/// e.g., the `visit_mod` method by default calls `visit::walk_mod`.
///
/// If you want to ensure that your code handles every variant
/// explicitly, you need to override each method. (And you also need
/// to monitor future changes to `Visitor` in case a new method with a
/// new default implementation gets introduced.)
pub trait Visitor<'ast>: Sized {
    fn visit_name(&mut self, _span: Span, _name: Symbol) {
        // Nothing to do.
    }
    fn visit_ident(&mut self, ident: Ident) {
        walk_ident(self, ident);
    }
    fn visit_mod(&mut self, m: &'ast Mod, _s: Span, _attrs: &[Attribute], _n: NodeId) {
        walk_mod(self, m);
    }
    fn visit_foreign_item(&mut self, i: &'ast ForeignItem) {
        walk_foreign_item(self, i)
    }
    fn visit_global_asm(&mut self, ga: &'ast GlobalAsm) {
        walk_global_asm(self, ga)
    }
    fn visit_item(&mut self, i: &'ast Item) {
        walk_item(self, i)
    }
    fn visit_local(&mut self, l: &'ast Local) {
        walk_local(self, l)
    }
    fn visit_block(&mut self, b: &'ast Block) {
        walk_block(self, b)
    }
    fn visit_stmt(&mut self, s: &'ast Stmt) {
        walk_stmt(self, s)
    }
    fn visit_param(&mut self, param: &'ast Param) {
        walk_param(self, param)
    }
    fn visit_arm(&mut self, a: &'ast Arm) {
        walk_arm(self, a)
    }
    fn visit_pat(&mut self, p: &'ast Pat) {
        walk_pat(self, p)
    }
    fn visit_anon_const(&mut self, c: &'ast AnonConst) {
        walk_anon_const(self, c)
    }
    fn visit_expr(&mut self, ex: &'ast Expr) {
        walk_expr(self, ex)
    }
    fn visit_expr_post(&mut self, _ex: &'ast Expr) {}
    fn visit_ty(&mut self, t: &'ast Ty) {
        walk_ty(self, t)
    }
    fn visit_generic_param(&mut self, param: &'ast GenericParam) {
        walk_generic_param(self, param)
    }
    fn visit_generics(&mut self, g: &'ast Generics) {
        walk_generics(self, g)
    }
    fn visit_where_predicate(&mut self, p: &'ast WherePredicate) {
        walk_where_predicate(self, p)
    }
    fn visit_fn(&mut self, fk: FnKind<'ast>, s: Span, _: NodeId) {
        walk_fn(self, fk, s)
    }
    fn visit_assoc_item(&mut self, i: &'ast AssocItem, ctxt: AssocCtxt) {
        walk_assoc_item(self, i, ctxt)
    }
    fn visit_trait_ref(&mut self, t: &'ast TraitRef) {
        walk_trait_ref(self, t)
    }
    fn visit_param_bound(&mut self, bounds: &'ast GenericBound) {
        walk_param_bound(self, bounds)
    }
    fn visit_poly_trait_ref(&mut self, t: &'ast PolyTraitRef, m: &'ast TraitBoundModifier) {
        walk_poly_trait_ref(self, t, m)
    }
    fn visit_variant_data(&mut self, s: &'ast VariantData) {
        walk_struct_def(self, s)
    }
    fn visit_struct_field(&mut self, s: &'ast StructField) {
        walk_struct_field(self, s)
    }
    fn visit_enum_def(
        &mut self,
        enum_definition: &'ast EnumDef,
        generics: &'ast Generics,
        item_id: NodeId,
        _: Span,
    ) {
        walk_enum_def(self, enum_definition, generics, item_id)
    }
    fn visit_variant(&mut self, v: &'ast Variant) {
        walk_variant(self, v)
    }
    fn visit_label(&mut self, label: &'ast Label) {
        walk_label(self, label)
    }
    fn visit_lifetime(&mut self, lifetime: &'ast Lifetime) {
        walk_lifetime(self, lifetime)
    }
    fn visit_mac(&mut self, _mac: &'ast MacCall) {
        panic!("visit_mac disabled by default");
        // N.B., see note about macros above.
        // if you really want a visitor that
        // works on macros, use this
        // definition in your trait impl:
        // visit::walk_mac(self, _mac)
    }
    fn visit_mac_def(&mut self, _mac: &'ast MacroDef, _id: NodeId) {
        // Nothing to do
    }
    fn visit_path(&mut self, path: &'ast Path, _id: NodeId) {
        walk_path(self, path)
    }
    fn visit_use_tree(&mut self, use_tree: &'ast UseTree, id: NodeId, _nested: bool) {
        walk_use_tree(self, use_tree, id)
    }
    fn visit_path_segment(&mut self, path_span: Span, path_segment: &'ast PathSegment) {
        walk_path_segment(self, path_span, path_segment)
    }
    fn visit_generic_args(&mut self, path_span: Span, generic_args: &'ast GenericArgs) {
        walk_generic_args(self, path_span, generic_args)
    }
    fn visit_generic_arg(&mut self, generic_arg: &'ast GenericArg) {
        walk_generic_arg(self, generic_arg)
    }
    fn visit_assoc_ty_constraint(&mut self, constraint: &'ast AssocTyConstraint) {
        walk_assoc_ty_constraint(self, constraint)
    }
    fn visit_attribute(&mut self, attr: &'ast Attribute) {
        walk_attribute(self, attr)
    }
    fn visit_vis(&mut self, vis: &'ast Visibility) {
        walk_vis(self, vis)
    }
    fn visit_fn_ret_ty(&mut self, ret_ty: &'ast FnRetTy) {
        walk_fn_ret_ty(self, ret_ty)
    }
    fn visit_fn_header(&mut self, _header: &'ast FnHeader) {
        // Nothing to do
    }
    fn visit_field(&mut self, f: &'ast Field) {
        walk_field(self, f)
    }
    fn visit_field_pattern(&mut self, fp: &'ast FieldPat) {
        walk_field_pattern(self, fp)
    }
}

#[macro_export]
macro_rules! walk_list {
    ($visitor: expr, $method: ident, $list: expr) => {
        for elem in $list {
            $visitor.$method(elem)
        }
    };
    ($visitor: expr, $method: ident, $list: expr, $($extra_args: expr),*) => {
        for elem in $list {
            $visitor.$method(elem, $($extra_args,)*)
        }
    }
}

pub fn walk_ident<'a, V: Visitor<'a>>(visitor: &mut V, ident: Ident) {
    visitor.visit_name(ident.span, ident.name);
}

pub fn walk_crate<'a, V: Visitor<'a>>(visitor: &mut V, krate: &'a Crate) {
    visitor.visit_mod(&krate.module, krate.span, &krate.attrs, CRATE_NODE_ID);
    walk_list!(visitor, visit_attribute, &krate.attrs);
}

pub fn walk_mod<'a, V: Visitor<'a>>(visitor: &mut V, module: &'a Mod) {
    walk_list!(visitor, visit_item, &module.items);
}

pub fn walk_local<'a, V: Visitor<'a>>(visitor: &mut V, local: &'a Local) {
    for attr in local.attrs.iter() {
        visitor.visit_attribute(attr);
    }
    visitor.visit_pat(&local.pat);
    walk_list!(visitor, visit_ty, &local.ty);
    walk_list!(visitor, visit_expr, &local.init);
}

pub fn walk_label<'a, V: Visitor<'a>>(visitor: &mut V, label: &'a Label) {
    visitor.visit_ident(label.ident);
}

pub fn walk_lifetime<'a, V: Visitor<'a>>(visitor: &mut V, lifetime: &'a Lifetime) {
    visitor.visit_ident(lifetime.ident);
}

pub fn walk_poly_trait_ref<'a, V>(
    visitor: &mut V,
    trait_ref: &'a PolyTraitRef,
    _: &TraitBoundModifier,
) where
    V: Visitor<'a>,
{
    walk_list!(visitor, visit_generic_param, &trait_ref.bound_generic_params);
    visitor.visit_trait_ref(&trait_ref.trait_ref);
}

pub fn walk_trait_ref<'a, V: Visitor<'a>>(visitor: &mut V, trait_ref: &'a TraitRef) {
    visitor.visit_path(&trait_ref.path, trait_ref.ref_id)
}

pub fn walk_item<'a, V: Visitor<'a>>(visitor: &mut V, item: &'a Item) {
    visitor.visit_vis(&item.vis);
    visitor.visit_ident(item.ident);
    match item.kind {
        ItemKind::ExternCrate(orig_name) => {
            if let Some(orig_name) = orig_name {
                visitor.visit_name(item.span, orig_name);
            }
        }
        ItemKind::Use(ref use_tree) => visitor.visit_use_tree(use_tree, item.id, false),
        ItemKind::Static(ref typ, _, ref expr) | ItemKind::Const(_, ref typ, ref expr) => {
            visitor.visit_ty(typ);
            walk_list!(visitor, visit_expr, expr);
        }
        ItemKind::Fn(_, ref sig, ref generics, ref body) => {
            visitor.visit_generics(generics);
            let kind = FnKind::Fn(FnCtxt::Free, item.ident, sig, &item.vis, body.as_deref());
            visitor.visit_fn(kind, item.span, item.id)
        }
        ItemKind::Mod(ref module) => visitor.visit_mod(module, item.span, &item.attrs, item.id),
        ItemKind::ForeignMod(ref foreign_module) => {
            walk_list!(visitor, visit_foreign_item, &foreign_module.items);
        }
        ItemKind::GlobalAsm(ref ga) => visitor.visit_global_asm(ga),
        ItemKind::TyAlias(_, ref generics, ref bounds, ref ty) => {
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_param_bound, bounds);
            walk_list!(visitor, visit_ty, ty);
        }
        ItemKind::Enum(ref enum_definition, ref generics) => {
            visitor.visit_generics(generics);
            visitor.visit_enum_def(enum_definition, generics, item.id, item.span)
        }
        ItemKind::Impl {
            unsafety: _,
            polarity: _,
            defaultness: _,
            constness: _,
            ref generics,
            ref of_trait,
            ref self_ty,
            ref items,
        } => {
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_trait_ref, of_trait);
            visitor.visit_ty(self_ty);
            walk_list!(visitor, visit_assoc_item, items, AssocCtxt::Impl);
        }
        ItemKind::Struct(ref struct_definition, ref generics)
        | ItemKind::Union(ref struct_definition, ref generics) => {
            visitor.visit_generics(generics);
            visitor.visit_variant_data(struct_definition);
        }
        ItemKind::Trait(.., ref generics, ref bounds, ref items) => {
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_param_bound, bounds);
            walk_list!(visitor, visit_assoc_item, items, AssocCtxt::Trait);
        }
        ItemKind::TraitAlias(ref generics, ref bounds) => {
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_param_bound, bounds);
        }
        ItemKind::MacCall(ref mac) => visitor.visit_mac(mac),
        ItemKind::MacroDef(ref ts) => visitor.visit_mac_def(ts, item.id),
    }
    walk_list!(visitor, visit_attribute, &item.attrs);
}

pub fn walk_enum_def<'a, V: Visitor<'a>>(
    visitor: &mut V,
    enum_definition: &'a EnumDef,
    _: &'a Generics,
    _: NodeId,
) {
    walk_list!(visitor, visit_variant, &enum_definition.variants);
}

pub fn walk_variant<'a, V: Visitor<'a>>(visitor: &mut V, variant: &'a Variant)
where
    V: Visitor<'a>,
{
    visitor.visit_ident(variant.ident);
    visitor.visit_vis(&variant.vis);
    visitor.visit_variant_data(&variant.data);
    walk_list!(visitor, visit_anon_const, &variant.disr_expr);
    walk_list!(visitor, visit_attribute, &variant.attrs);
}

pub fn walk_field<'a, V: Visitor<'a>>(visitor: &mut V, f: &'a Field) {
    visitor.visit_expr(&f.expr);
    visitor.visit_ident(f.ident);
    walk_list!(visitor, visit_attribute, f.attrs.iter());
}

pub fn walk_field_pattern<'a, V: Visitor<'a>>(visitor: &mut V, fp: &'a FieldPat) {
    visitor.visit_ident(fp.ident);
    visitor.visit_pat(&fp.pat);
    walk_list!(visitor, visit_attribute, fp.attrs.iter());
}

pub fn walk_ty<'a, V: Visitor<'a>>(visitor: &mut V, typ: &'a Ty) {
    match typ.kind {
        TyKind::Slice(ref ty) | TyKind::Paren(ref ty) => visitor.visit_ty(ty),
        TyKind::Ptr(ref mutable_type) => visitor.visit_ty(&mutable_type.ty),
        TyKind::Rptr(ref opt_lifetime, ref mutable_type) => {
            walk_list!(visitor, visit_lifetime, opt_lifetime);
            visitor.visit_ty(&mutable_type.ty)
        }
        TyKind::Tup(ref tuple_element_types) => {
            walk_list!(visitor, visit_ty, tuple_element_types);
        }
        TyKind::BareFn(ref function_declaration) => {
            walk_list!(visitor, visit_generic_param, &function_declaration.generic_params);
            walk_fn_decl(visitor, &function_declaration.decl);
        }
        TyKind::Path(ref maybe_qself, ref path) => {
            if let Some(ref qself) = *maybe_qself {
                visitor.visit_ty(&qself.ty);
            }
            visitor.visit_path(path, typ.id);
        }
        TyKind::Array(ref ty, ref length) => {
            visitor.visit_ty(ty);
            visitor.visit_anon_const(length)
        }
        TyKind::TraitObject(ref bounds, ..) | TyKind::ImplTrait(_, ref bounds) => {
            walk_list!(visitor, visit_param_bound, bounds);
        }
        TyKind::Typeof(ref expression) => visitor.visit_anon_const(expression),
        TyKind::Infer | TyKind::ImplicitSelf | TyKind::Err => {}
        TyKind::MacCall(ref mac) => visitor.visit_mac(mac),
        TyKind::Never | TyKind::CVarArgs => {}
    }
}

pub fn walk_path<'a, V: Visitor<'a>>(visitor: &mut V, path: &'a Path) {
    for segment in &path.segments {
        visitor.visit_path_segment(path.span, segment);
    }
}

pub fn walk_use_tree<'a, V: Visitor<'a>>(visitor: &mut V, use_tree: &'a UseTree, id: NodeId) {
    visitor.visit_path(&use_tree.prefix, id);
    match use_tree.kind {
        UseTreeKind::Simple(rename, ..) => {
            // The extra IDs are handled during HIR lowering.
            if let Some(rename) = rename {
                visitor.visit_ident(rename);
            }
        }
        UseTreeKind::Glob => {}
        UseTreeKind::Nested(ref use_trees) => {
            for &(ref nested_tree, nested_id) in use_trees {
                visitor.visit_use_tree(nested_tree, nested_id, true);
            }
        }
    }
}

pub fn walk_path_segment<'a, V: Visitor<'a>>(
    visitor: &mut V,
    path_span: Span,
    segment: &'a PathSegment,
) {
    visitor.visit_ident(segment.ident);
    if let Some(ref args) = segment.args {
        visitor.visit_generic_args(path_span, args);
    }
}

pub fn walk_generic_args<'a, V>(visitor: &mut V, _path_span: Span, generic_args: &'a GenericArgs)
where
    V: Visitor<'a>,
{
    match *generic_args {
        GenericArgs::AngleBracketed(ref data) => {
            for arg in &data.args {
                match arg {
                    AngleBracketedArg::Arg(a) => visitor.visit_generic_arg(a),
                    AngleBracketedArg::Constraint(c) => visitor.visit_assoc_ty_constraint(c),
                }
            }
        }
        GenericArgs::Parenthesized(ref data) => {
            walk_list!(visitor, visit_ty, &data.inputs);
            walk_fn_ret_ty(visitor, &data.output);
        }
    }
}

pub fn walk_generic_arg<'a, V>(visitor: &mut V, generic_arg: &'a GenericArg)
where
    V: Visitor<'a>,
{
    match generic_arg {
        GenericArg::Lifetime(lt) => visitor.visit_lifetime(lt),
        GenericArg::Type(ty) => visitor.visit_ty(ty),
        GenericArg::Const(ct) => visitor.visit_anon_const(ct),
    }
}

pub fn walk_assoc_ty_constraint<'a, V: Visitor<'a>>(
    visitor: &mut V,
    constraint: &'a AssocTyConstraint,
) {
    visitor.visit_ident(constraint.ident);
    match constraint.kind {
        AssocTyConstraintKind::Equality { ref ty } => {
            visitor.visit_ty(ty);
        }
        AssocTyConstraintKind::Bound { ref bounds } => {
            walk_list!(visitor, visit_param_bound, bounds);
        }
    }
}

pub fn walk_pat<'a, V: Visitor<'a>>(visitor: &mut V, pattern: &'a Pat) {
    match pattern.kind {
        PatKind::TupleStruct(ref path, ref elems) => {
            visitor.visit_path(path, pattern.id);
            walk_list!(visitor, visit_pat, elems);
        }
        PatKind::Path(ref opt_qself, ref path) => {
            if let Some(ref qself) = *opt_qself {
                visitor.visit_ty(&qself.ty);
            }
            visitor.visit_path(path, pattern.id)
        }
        PatKind::Struct(ref path, ref fields, _) => {
            visitor.visit_path(path, pattern.id);
            walk_list!(visitor, visit_field_pattern, fields);
        }
        PatKind::Box(ref subpattern)
        | PatKind::Ref(ref subpattern, _)
        | PatKind::Paren(ref subpattern) => visitor.visit_pat(subpattern),
        PatKind::Ident(_, ident, ref optional_subpattern) => {
            visitor.visit_ident(ident);
            walk_list!(visitor, visit_pat, optional_subpattern);
        }
        PatKind::Lit(ref expression) => visitor.visit_expr(expression),
        PatKind::Range(ref lower_bound, ref upper_bound, _) => {
            walk_list!(visitor, visit_expr, lower_bound);
            walk_list!(visitor, visit_expr, upper_bound);
        }
        PatKind::Wild | PatKind::Rest => {}
        PatKind::Tuple(ref elems) | PatKind::Slice(ref elems) | PatKind::Or(ref elems) => {
            walk_list!(visitor, visit_pat, elems);
        }
        PatKind::MacCall(ref mac) => visitor.visit_mac(mac),
    }
}

pub fn walk_foreign_item<'a, V: Visitor<'a>>(visitor: &mut V, item: &'a ForeignItem) {
    let Item { id, span, ident, ref vis, ref attrs, ref kind, tokens: _ } = *item;
    visitor.visit_vis(vis);
    visitor.visit_ident(ident);
    walk_list!(visitor, visit_attribute, attrs);
    match kind {
        ForeignItemKind::Static(ty, _, expr) => {
            visitor.visit_ty(ty);
            walk_list!(visitor, visit_expr, expr);
        }
        ForeignItemKind::Fn(_, sig, generics, body) => {
            visitor.visit_generics(generics);
            let kind = FnKind::Fn(FnCtxt::Foreign, ident, sig, vis, body.as_deref());
            visitor.visit_fn(kind, span, id);
        }
        ForeignItemKind::TyAlias(_, generics, bounds, ty) => {
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_param_bound, bounds);
            walk_list!(visitor, visit_ty, ty);
        }
        ForeignItemKind::MacCall(mac) => {
            visitor.visit_mac(mac);
        }
    }
}

pub fn walk_global_asm<'a, V: Visitor<'a>>(_: &mut V, _: &'a GlobalAsm) {
    // Empty!
}

pub fn walk_param_bound<'a, V: Visitor<'a>>(visitor: &mut V, bound: &'a GenericBound) {
    match *bound {
        GenericBound::Trait(ref typ, ref modifier) => visitor.visit_poly_trait_ref(typ, modifier),
        GenericBound::Outlives(ref lifetime) => visitor.visit_lifetime(lifetime),
    }
}

pub fn walk_generic_param<'a, V: Visitor<'a>>(visitor: &mut V, param: &'a GenericParam) {
    visitor.visit_ident(param.ident);
    walk_list!(visitor, visit_attribute, param.attrs.iter());
    walk_list!(visitor, visit_param_bound, &param.bounds);
    match param.kind {
        GenericParamKind::Lifetime => (),
        GenericParamKind::Type { ref default } => walk_list!(visitor, visit_ty, default),
        GenericParamKind::Const { ref ty, .. } => visitor.visit_ty(ty),
    }
}

pub fn walk_generics<'a, V: Visitor<'a>>(visitor: &mut V, generics: &'a Generics) {
    walk_list!(visitor, visit_generic_param, &generics.params);
    walk_list!(visitor, visit_where_predicate, &generics.where_clause.predicates);
}

pub fn walk_where_predicate<'a, V: Visitor<'a>>(visitor: &mut V, predicate: &'a WherePredicate) {
    match *predicate {
        WherePredicate::BoundPredicate(WhereBoundPredicate {
            ref bounded_ty,
            ref bounds,
            ref bound_generic_params,
            ..
        }) => {
            visitor.visit_ty(bounded_ty);
            walk_list!(visitor, visit_param_bound, bounds);
            walk_list!(visitor, visit_generic_param, bound_generic_params);
        }
        WherePredicate::RegionPredicate(WhereRegionPredicate {
            ref lifetime, ref bounds, ..
        }) => {
            visitor.visit_lifetime(lifetime);
            walk_list!(visitor, visit_param_bound, bounds);
        }
        WherePredicate::EqPredicate(WhereEqPredicate { ref lhs_ty, ref rhs_ty, .. }) => {
            visitor.visit_ty(lhs_ty);
            visitor.visit_ty(rhs_ty);
        }
    }
}

pub fn walk_fn_ret_ty<'a, V: Visitor<'a>>(visitor: &mut V, ret_ty: &'a FnRetTy) {
    if let FnRetTy::Ty(ref output_ty) = *ret_ty {
        visitor.visit_ty(output_ty)
    }
}

pub fn walk_fn_decl<'a, V: Visitor<'a>>(visitor: &mut V, function_declaration: &'a FnDecl) {
    for param in &function_declaration.inputs {
        visitor.visit_param(param);
    }
    visitor.visit_fn_ret_ty(&function_declaration.output);
}

pub fn walk_fn<'a, V: Visitor<'a>>(visitor: &mut V, kind: FnKind<'a>, _span: Span) {
    match kind {
        FnKind::Fn(_, _, sig, _, body) => {
            visitor.visit_fn_header(&sig.header);
            walk_fn_decl(visitor, &sig.decl);
            walk_list!(visitor, visit_block, body);
        }
        FnKind::Closure(decl, body) => {
            walk_fn_decl(visitor, decl);
            visitor.visit_expr(body);
        }
    }
}

pub fn walk_assoc_item<'a, V: Visitor<'a>>(visitor: &mut V, item: &'a AssocItem, ctxt: AssocCtxt) {
    let Item { id, span, ident, ref vis, ref attrs, ref kind, tokens: _ } = *item;
    visitor.visit_vis(vis);
    visitor.visit_ident(ident);
    walk_list!(visitor, visit_attribute, attrs);
    match kind {
        AssocItemKind::Const(_, ty, expr) => {
            visitor.visit_ty(ty);
            walk_list!(visitor, visit_expr, expr);
        }
        AssocItemKind::Fn(_, sig, generics, body) => {
            visitor.visit_generics(generics);
            let kind = FnKind::Fn(FnCtxt::Assoc(ctxt), ident, sig, vis, body.as_deref());
            visitor.visit_fn(kind, span, id);
        }
        AssocItemKind::TyAlias(_, generics, bounds, ty) => {
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_param_bound, bounds);
            walk_list!(visitor, visit_ty, ty);
        }
        AssocItemKind::MacCall(mac) => {
            visitor.visit_mac(mac);
        }
    }
}

pub fn walk_struct_def<'a, V: Visitor<'a>>(visitor: &mut V, struct_definition: &'a VariantData) {
    walk_list!(visitor, visit_struct_field, struct_definition.fields());
}

pub fn walk_struct_field<'a, V: Visitor<'a>>(visitor: &mut V, struct_field: &'a StructField) {
    visitor.visit_vis(&struct_field.vis);
    if let Some(ident) = struct_field.ident {
        visitor.visit_ident(ident);
    }
    visitor.visit_ty(&struct_field.ty);
    walk_list!(visitor, visit_attribute, &struct_field.attrs);
}

pub fn walk_block<'a, V: Visitor<'a>>(visitor: &mut V, block: &'a Block) {
    walk_list!(visitor, visit_stmt, &block.stmts);
}

pub fn walk_stmt<'a, V: Visitor<'a>>(visitor: &mut V, statement: &'a Stmt) {
    match statement.kind {
        StmtKind::Local(ref local) => visitor.visit_local(local),
        StmtKind::Item(ref item) => visitor.visit_item(item),
        StmtKind::Expr(ref expr) | StmtKind::Semi(ref expr) => visitor.visit_expr(expr),
        StmtKind::Empty => {}
        StmtKind::MacCall(ref mac) => {
            let MacCallStmt { ref mac, style: _, ref attrs } = **mac;
            visitor.visit_mac(mac);
            for attr in attrs.iter() {
                visitor.visit_attribute(attr);
            }
        }
    }
}

pub fn walk_mac<'a, V: Visitor<'a>>(visitor: &mut V, mac: &'a MacCall) {
    visitor.visit_path(&mac.path, DUMMY_NODE_ID);
}

pub fn walk_anon_const<'a, V: Visitor<'a>>(visitor: &mut V, constant: &'a AnonConst) {
    visitor.visit_expr(&constant.value);
}

pub fn walk_expr<'a, V: Visitor<'a>>(visitor: &mut V, expression: &'a Expr) {
    walk_list!(visitor, visit_attribute, expression.attrs.iter());

    match expression.kind {
        ExprKind::Box(ref subexpression) => visitor.visit_expr(subexpression),
        ExprKind::Array(ref subexpressions) => {
            walk_list!(visitor, visit_expr, subexpressions);
        }
        ExprKind::ConstBlock(ref anon_const) => visitor.visit_anon_const(anon_const),
        ExprKind::Repeat(ref element, ref count) => {
            visitor.visit_expr(element);
            visitor.visit_anon_const(count)
        }
        ExprKind::Struct(ref path, ref fields, ref optional_base) => {
            visitor.visit_path(path, expression.id);
            walk_list!(visitor, visit_field, fields);
            walk_list!(visitor, visit_expr, optional_base);
        }
        ExprKind::Tup(ref subexpressions) => {
            walk_list!(visitor, visit_expr, subexpressions);
        }
        ExprKind::Call(ref callee_expression, ref arguments) => {
            visitor.visit_expr(callee_expression);
            walk_list!(visitor, visit_expr, arguments);
        }
        ExprKind::MethodCall(ref segment, ref arguments, _span) => {
            visitor.visit_path_segment(expression.span, segment);
            walk_list!(visitor, visit_expr, arguments);
        }
        ExprKind::Binary(_, ref left_expression, ref right_expression) => {
            visitor.visit_expr(left_expression);
            visitor.visit_expr(right_expression)
        }
        ExprKind::AddrOf(_, _, ref subexpression) | ExprKind::Unary(_, ref subexpression) => {
            visitor.visit_expr(subexpression)
        }
        ExprKind::Cast(ref subexpression, ref typ) | ExprKind::Type(ref subexpression, ref typ) => {
            visitor.visit_expr(subexpression);
            visitor.visit_ty(typ)
        }
        ExprKind::Let(ref pat, ref scrutinee) => {
            visitor.visit_pat(pat);
            visitor.visit_expr(scrutinee);
        }
        ExprKind::If(ref head_expression, ref if_block, ref optional_else) => {
            visitor.visit_expr(head_expression);
            visitor.visit_block(if_block);
            walk_list!(visitor, visit_expr, optional_else);
        }
        ExprKind::While(ref subexpression, ref block, ref opt_label) => {
            walk_list!(visitor, visit_label, opt_label);
            visitor.visit_expr(subexpression);
            visitor.visit_block(block);
        }
        ExprKind::ForLoop(ref pattern, ref subexpression, ref block, ref opt_label) => {
            walk_list!(visitor, visit_label, opt_label);
            visitor.visit_pat(pattern);
            visitor.visit_expr(subexpression);
            visitor.visit_block(block);
        }
        ExprKind::Loop(ref block, ref opt_label) => {
            walk_list!(visitor, visit_label, opt_label);
            visitor.visit_block(block);
        }
        ExprKind::Match(ref subexpression, ref arms) => {
            visitor.visit_expr(subexpression);
            walk_list!(visitor, visit_arm, arms);
        }
        ExprKind::Closure(_, _, _, ref decl, ref body, _decl_span) => {
            visitor.visit_fn(FnKind::Closure(decl, body), expression.span, expression.id)
        }
        ExprKind::Block(ref block, ref opt_label) => {
            walk_list!(visitor, visit_label, opt_label);
            visitor.visit_block(block);
        }
        ExprKind::Async(_, _, ref body) => {
            visitor.visit_block(body);
        }
        ExprKind::Await(ref expr) => visitor.visit_expr(expr),
        ExprKind::Assign(ref lhs, ref rhs, _) => {
            visitor.visit_expr(lhs);
            visitor.visit_expr(rhs);
        }
        ExprKind::AssignOp(_, ref left_expression, ref right_expression) => {
            visitor.visit_expr(left_expression);
            visitor.visit_expr(right_expression);
        }
        ExprKind::Field(ref subexpression, ident) => {
            visitor.visit_expr(subexpression);
            visitor.visit_ident(ident);
        }
        ExprKind::Index(ref main_expression, ref index_expression) => {
            visitor.visit_expr(main_expression);
            visitor.visit_expr(index_expression)
        }
        ExprKind::Range(ref start, ref end, _) => {
            walk_list!(visitor, visit_expr, start);
            walk_list!(visitor, visit_expr, end);
        }
        ExprKind::Path(ref maybe_qself, ref path) => {
            if let Some(ref qself) = *maybe_qself {
                visitor.visit_ty(&qself.ty);
            }
            visitor.visit_path(path, expression.id)
        }
        ExprKind::Break(ref opt_label, ref opt_expr) => {
            walk_list!(visitor, visit_label, opt_label);
            walk_list!(visitor, visit_expr, opt_expr);
        }
        ExprKind::Continue(ref opt_label) => {
            walk_list!(visitor, visit_label, opt_label);
        }
        ExprKind::Ret(ref optional_expression) => {
            walk_list!(visitor, visit_expr, optional_expression);
        }
        ExprKind::MacCall(ref mac) => visitor.visit_mac(mac),
        ExprKind::Paren(ref subexpression) => visitor.visit_expr(subexpression),
        ExprKind::InlineAsm(ref ia) => {
            for (op, _) in &ia.operands {
                match op {
                    InlineAsmOperand::In { expr, .. }
                    | InlineAsmOperand::InOut { expr, .. }
                    | InlineAsmOperand::Const { expr, .. }
                    | InlineAsmOperand::Sym { expr, .. } => visitor.visit_expr(expr),
                    InlineAsmOperand::Out { expr, .. } => {
                        if let Some(expr) = expr {
                            visitor.visit_expr(expr);
                        }
                    }
                    InlineAsmOperand::SplitInOut { in_expr, out_expr, .. } => {
                        visitor.visit_expr(in_expr);
                        if let Some(out_expr) = out_expr {
                            visitor.visit_expr(out_expr);
                        }
                    }
                }
            }
        }
        ExprKind::LlvmInlineAsm(ref ia) => {
            for &(_, ref input) in &ia.inputs {
                visitor.visit_expr(input)
            }
            for output in &ia.outputs {
                visitor.visit_expr(&output.expr)
            }
        }
        ExprKind::Yield(ref optional_expression) => {
            walk_list!(visitor, visit_expr, optional_expression);
        }
        ExprKind::Try(ref subexpression) => visitor.visit_expr(subexpression),
        ExprKind::TryBlock(ref body) => visitor.visit_block(body),
        ExprKind::Lit(_) | ExprKind::Err => {}
    }

    visitor.visit_expr_post(expression)
}

pub fn walk_param<'a, V: Visitor<'a>>(visitor: &mut V, param: &'a Param) {
    walk_list!(visitor, visit_attribute, param.attrs.iter());
    visitor.visit_pat(&param.pat);
    visitor.visit_ty(&param.ty);
}

pub fn walk_arm<'a, V: Visitor<'a>>(visitor: &mut V, arm: &'a Arm) {
    visitor.visit_pat(&arm.pat);
    walk_list!(visitor, visit_expr, &arm.guard);
    visitor.visit_expr(&arm.body);
    walk_list!(visitor, visit_attribute, &arm.attrs);
}

pub fn walk_vis<'a, V: Visitor<'a>>(visitor: &mut V, vis: &'a Visibility) {
    if let VisibilityKind::Restricted { ref path, id } = vis.kind {
        visitor.visit_path(path, id);
    }
}

pub fn walk_attribute<'a, V: Visitor<'a>>(visitor: &mut V, attr: &'a Attribute) {
    match attr.kind {
        AttrKind::Normal(ref item) => walk_mac_args(visitor, &item.args),
        AttrKind::DocComment(..) => {}
    }
}

pub fn walk_mac_args<'a, V: Visitor<'a>>(visitor: &mut V, args: &'a MacArgs) {
    match args {
        MacArgs::Empty => {}
        MacArgs::Delimited(_dspan, _delim, _tokens) => {}
        // The value in `#[key = VALUE]` must be visited as an expression for backward
        // compatibility, so that macros can be expanded in that position.
        MacArgs::Eq(_eq_span, tokens) => match tokens.trees_ref().next() {
            Some(TokenTree::Token(token)) => match &token.kind {
                token::Interpolated(nt) => match &**nt {
                    token::NtExpr(expr) => visitor.visit_expr(expr),
                    t => panic!("unexpected token in key-value attribute: {:?}", t),
                },
                token::Literal(..) | token::Ident(..) => {}
                t => panic!("unexpected token in key-value attribute: {:?}", t),
            },
            t => panic!("unexpected token in key-value attribute: {:?}", t),
        },
    }
}
