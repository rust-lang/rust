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
use crate::parse::token::Token;
use crate::tokenstream::{TokenTree, TokenStream};

use syntax_pos::Span;

#[derive(Copy, Clone)]
pub enum FnKind<'a> {
    /// fn foo() or extern "Abi" fn foo()
    ItemFn(Ident, &'a FnHeader, &'a Visibility, &'a Block),

    /// fn foo(&self)
    Method(Ident, &'a MethodSig, Option<&'a Visibility>, &'a Block),

    /// |x, y| body
    Closure(&'a Expr),
}

impl<'a> FnKind<'a> {
    pub fn header(&self) -> Option<&'a FnHeader> {
        match *self {
            FnKind::ItemFn(_, header, _, _) => Some(header),
            FnKind::Method(_, sig, _, _) => Some(&sig.header),
            FnKind::Closure(_) => None,
        }
    }
}

/// Each method of the Visitor trait is a hook to be potentially
/// overridden. Each method's default implementation recursively visits
/// the substructure of the input via the corresponding `walk` method;
/// e.g., the `visit_mod` method by default calls `visit::walk_mod`.
///
/// If you want to ensure that your code handles every variant
/// explicitly, you need to override each method. (And you also need
/// to monitor future changes to `Visitor` in case a new method with a
/// new default implementation gets introduced.)
pub trait Visitor<'ast>: Sized {
    fn visit_name(&mut self, _span: Span, _name: Name) {
        // Nothing to do.
    }
    fn visit_ident(&mut self, ident: Ident) {
        walk_ident(self, ident);
    }
    fn visit_mod(&mut self, m: &'ast Mod, _s: Span, _attrs: &[Attribute], _n: NodeId) {
        walk_mod(self, m);
    }
    fn visit_foreign_item(&mut self, i: &'ast ForeignItem) { walk_foreign_item(self, i) }
    fn visit_global_asm(&mut self, ga: &'ast GlobalAsm) { walk_global_asm(self, ga) }
    fn visit_item(&mut self, i: &'ast Item) { walk_item(self, i) }
    fn visit_local(&mut self, l: &'ast Local) { walk_local(self, l) }
    fn visit_block(&mut self, b: &'ast Block) { walk_block(self, b) }
    fn visit_stmt(&mut self, s: &'ast Stmt) { walk_stmt(self, s) }
    fn visit_arm(&mut self, a: &'ast Arm) { walk_arm(self, a) }
    fn visit_pat(&mut self, p: &'ast Pat) { walk_pat(self, p) }
    fn visit_anon_const(&mut self, c: &'ast AnonConst) { walk_anon_const(self, c) }
    fn visit_expr(&mut self, ex: &'ast Expr) { walk_expr(self, ex) }
    fn visit_expr_post(&mut self, _ex: &'ast Expr) { }
    fn visit_ty(&mut self, t: &'ast Ty) { walk_ty(self, t) }
    fn visit_generic_param(&mut self, param: &'ast GenericParam) {
        walk_generic_param(self, param)
    }
    fn visit_generics(&mut self, g: &'ast Generics) { walk_generics(self, g) }
    fn visit_where_predicate(&mut self, p: &'ast WherePredicate) {
        walk_where_predicate(self, p)
    }
    fn visit_fn(&mut self, fk: FnKind<'ast>, fd: &'ast FnDecl, s: Span, _: NodeId) {
        walk_fn(self, fk, fd, s)
    }
    fn visit_trait_item(&mut self, ti: &'ast TraitItem) { walk_trait_item(self, ti) }
    fn visit_impl_item(&mut self, ii: &'ast ImplItem) { walk_impl_item(self, ii) }
    fn visit_trait_ref(&mut self, t: &'ast TraitRef) { walk_trait_ref(self, t) }
    fn visit_param_bound(&mut self, bounds: &'ast GenericBound) {
        walk_param_bound(self, bounds)
    }
    fn visit_poly_trait_ref(&mut self, t: &'ast PolyTraitRef, m: &'ast TraitBoundModifier) {
        walk_poly_trait_ref(self, t, m)
    }
    fn visit_variant_data(&mut self, s: &'ast VariantData, _: Ident,
                          _: &'ast Generics, _: NodeId, _: Span) {
        walk_struct_def(self, s)
    }
    fn visit_struct_field(&mut self, s: &'ast StructField) { walk_struct_field(self, s) }
    fn visit_enum_def(&mut self, enum_definition: &'ast EnumDef,
                      generics: &'ast Generics, item_id: NodeId, _: Span) {
        walk_enum_def(self, enum_definition, generics, item_id)
    }
    fn visit_variant(&mut self, v: &'ast Variant, g: &'ast Generics, item_id: NodeId) {
        walk_variant(self, v, g, item_id)
    }
    fn visit_label(&mut self, label: &'ast Label) {
        walk_label(self, label)
    }
    fn visit_lifetime(&mut self, lifetime: &'ast Lifetime) {
        walk_lifetime(self, lifetime)
    }
    fn visit_mac(&mut self, _mac: &'ast Mac) {
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
        match generic_arg {
            GenericArg::Lifetime(lt) => self.visit_lifetime(lt),
            GenericArg::Type(ty) => self.visit_ty(ty),
            GenericArg::Const(ct) => self.visit_anon_const(ct),
        }
    }
    fn visit_assoc_ty_constraint(&mut self, constraint: &'ast AssocTyConstraint) {
        walk_assoc_ty_constraint(self, constraint)
    }
    fn visit_attribute(&mut self, attr: &'ast Attribute) {
        walk_attribute(self, attr)
    }
    fn visit_tt(&mut self, tt: TokenTree) {
        walk_tt(self, tt)
    }
    fn visit_tts(&mut self, tts: TokenStream) {
        walk_tts(self, tts)
    }
    fn visit_token(&mut self, _t: Token) {}
    // FIXME: add `visit_interpolated` and `walk_interpolated`
    fn visit_vis(&mut self, vis: &'ast Visibility) {
        walk_vis(self, vis)
    }
    fn visit_fn_ret_ty(&mut self, ret_ty: &'ast FunctionRetTy) {
        walk_fn_ret_ty(self, ret_ty)
    }
    fn visit_fn_header(&mut self, _header: &'ast FnHeader) {
        // Nothing to do
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

pub fn walk_poly_trait_ref<'a, V>(visitor: &mut V,
                                  trait_ref: &'a PolyTraitRef,
                                  _: &TraitBoundModifier)
    where V: Visitor<'a>,
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
    match item.node {
        ItemKind::ExternCrate(orig_name) => {
            if let Some(orig_name) = orig_name {
                visitor.visit_name(item.span, orig_name);
            }
        }
        ItemKind::Use(ref use_tree) => {
            visitor.visit_use_tree(use_tree, item.id, false)
        }
        ItemKind::Static(ref typ, _, ref expr) |
        ItemKind::Const(ref typ, ref expr) => {
            visitor.visit_ty(typ);
            visitor.visit_expr(expr);
        }
        ItemKind::Fn(ref declaration, ref header, ref generics, ref body) => {
            visitor.visit_generics(generics);
            visitor.visit_fn_header(header);
            visitor.visit_fn(FnKind::ItemFn(item.ident, header,
                                            &item.vis, body),
                             declaration,
                             item.span,
                             item.id)
        }
        ItemKind::Mod(ref module) => {
            visitor.visit_mod(module, item.span, &item.attrs, item.id)
        }
        ItemKind::ForeignMod(ref foreign_module) => {
            walk_list!(visitor, visit_foreign_item, &foreign_module.items);
        }
        ItemKind::GlobalAsm(ref ga) => visitor.visit_global_asm(ga),
        ItemKind::Ty(ref typ, ref generics) => {
            visitor.visit_ty(typ);
            visitor.visit_generics(generics)
        }
        ItemKind::Existential(ref bounds, ref generics) => {
            walk_list!(visitor, visit_param_bound, bounds);
            visitor.visit_generics(generics)
        }
        ItemKind::Enum(ref enum_definition, ref generics) => {
            visitor.visit_generics(generics);
            visitor.visit_enum_def(enum_definition, generics, item.id, item.span)
        }
        ItemKind::Impl(_, _, _,
                 ref generics,
                 ref opt_trait_reference,
                 ref typ,
                 ref impl_items) => {
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_trait_ref, opt_trait_reference);
            visitor.visit_ty(typ);
            walk_list!(visitor, visit_impl_item, impl_items);
        }
        ItemKind::Struct(ref struct_definition, ref generics) |
        ItemKind::Union(ref struct_definition, ref generics) => {
            visitor.visit_generics(generics);
            visitor.visit_variant_data(struct_definition, item.ident,
                                     generics, item.id, item.span);
        }
        ItemKind::Trait(.., ref generics, ref bounds, ref methods) => {
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_param_bound, bounds);
            walk_list!(visitor, visit_trait_item, methods);
        }
        ItemKind::TraitAlias(ref generics, ref bounds) => {
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_param_bound, bounds);
        }
        ItemKind::Mac(ref mac) => visitor.visit_mac(mac),
        ItemKind::MacroDef(ref ts) => visitor.visit_mac_def(ts, item.id),
    }
    walk_list!(visitor, visit_attribute, &item.attrs);
}

pub fn walk_enum_def<'a, V: Visitor<'a>>(visitor: &mut V,
                                 enum_definition: &'a EnumDef,
                                 generics: &'a Generics,
                                 item_id: NodeId) {
    walk_list!(visitor, visit_variant, &enum_definition.variants, generics, item_id);
}

pub fn walk_variant<'a, V>(visitor: &mut V,
                           variant: &'a Variant,
                           generics: &'a Generics,
                           item_id: NodeId)
    where V: Visitor<'a>,
{
    visitor.visit_ident(variant.node.ident);
    visitor.visit_variant_data(&variant.node.data, variant.node.ident,
                             generics, item_id, variant.span);
    walk_list!(visitor, visit_anon_const, &variant.node.disr_expr);
    walk_list!(visitor, visit_attribute, &variant.node.attrs);
}

pub fn walk_ty<'a, V: Visitor<'a>>(visitor: &mut V, typ: &'a Ty) {
    match typ.node {
        TyKind::Slice(ref ty) | TyKind::Paren(ref ty) => {
            visitor.visit_ty(ty)
        }
        TyKind::Ptr(ref mutable_type) => {
            visitor.visit_ty(&mutable_type.ty)
        }
        TyKind::Rptr(ref opt_lifetime, ref mutable_type) => {
            walk_list!(visitor, visit_lifetime, opt_lifetime);
            visitor.visit_ty(&mutable_type.ty)
        }
        TyKind::Never | TyKind::CVarArgs => {}
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
        TyKind::TraitObject(ref bounds, ..) |
        TyKind::ImplTrait(_, ref bounds) => {
            walk_list!(visitor, visit_param_bound, bounds);
        }
        TyKind::Typeof(ref expression) => {
            visitor.visit_anon_const(expression)
        }
        TyKind::Infer | TyKind::ImplicitSelf | TyKind::Err => {}
        TyKind::Mac(ref mac) => {
            visitor.visit_mac(mac)
        }
    }
}

pub fn walk_path<'a, V: Visitor<'a>>(visitor: &mut V, path: &'a Path) {
    for segment in &path.segments {
        visitor.visit_path_segment(path.span, segment);
    }
}

pub fn walk_use_tree<'a, V: Visitor<'a>>(
    visitor: &mut V, use_tree: &'a UseTree, id: NodeId,
) {
    visitor.visit_path(&use_tree.prefix, id);
    match use_tree.kind {
        UseTreeKind::Simple(rename, ..) => {
            // the extra IDs are handled during HIR lowering
            if let Some(rename) = rename {
                visitor.visit_ident(rename);
            }
        }
        UseTreeKind::Glob => {},
        UseTreeKind::Nested(ref use_trees) => {
            for &(ref nested_tree, nested_id) in use_trees {
                visitor.visit_use_tree(nested_tree, nested_id, true);
            }
        }
    }
}

pub fn walk_path_segment<'a, V: Visitor<'a>>(visitor: &mut V,
                                             path_span: Span,
                                             segment: &'a PathSegment) {
    visitor.visit_ident(segment.ident);
    if let Some(ref args) = segment.args {
        visitor.visit_generic_args(path_span, args);
    }
}

pub fn walk_generic_args<'a, V>(visitor: &mut V,
                                _path_span: Span,
                                generic_args: &'a GenericArgs)
    where V: Visitor<'a>,
{
    match *generic_args {
        GenericArgs::AngleBracketed(ref data) => {
            walk_list!(visitor, visit_generic_arg, &data.args);
            walk_list!(visitor, visit_assoc_ty_constraint, &data.constraints);
        }
        GenericArgs::Parenthesized(ref data) => {
            walk_list!(visitor, visit_ty, &data.inputs);
            walk_list!(visitor, visit_ty, &data.output);
        }
    }
}

pub fn walk_assoc_ty_constraint<'a, V: Visitor<'a>>(visitor: &mut V,
                                                    constraint: &'a AssocTyConstraint) {
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
    match pattern.node {
        PatKind::TupleStruct(ref path, ref children, _) => {
            visitor.visit_path(path, pattern.id);
            walk_list!(visitor, visit_pat, children);
        }
        PatKind::Path(ref opt_qself, ref path) => {
            if let Some(ref qself) = *opt_qself {
                visitor.visit_ty(&qself.ty);
            }
            visitor.visit_path(path, pattern.id)
        }
        PatKind::Struct(ref path, ref fields, _) => {
            visitor.visit_path(path, pattern.id);
            for field in fields {
                walk_list!(visitor, visit_attribute, field.node.attrs.iter());
                visitor.visit_ident(field.node.ident);
                visitor.visit_pat(&field.node.pat)
            }
        }
        PatKind::Tuple(ref tuple_elements, _) => {
            walk_list!(visitor, visit_pat, tuple_elements);
        }
        PatKind::Box(ref subpattern) |
        PatKind::Ref(ref subpattern, _) |
        PatKind::Paren(ref subpattern) => {
            visitor.visit_pat(subpattern)
        }
        PatKind::Ident(_, ident, ref optional_subpattern) => {
            visitor.visit_ident(ident);
            walk_list!(visitor, visit_pat, optional_subpattern);
        }
        PatKind::Lit(ref expression) => visitor.visit_expr(expression),
        PatKind::Range(ref lower_bound, ref upper_bound, _) => {
            visitor.visit_expr(lower_bound);
            visitor.visit_expr(upper_bound);
        }
        PatKind::Wild => (),
        PatKind::Slice(ref prepatterns, ref slice_pattern, ref postpatterns) => {
            walk_list!(visitor, visit_pat, prepatterns);
            walk_list!(visitor, visit_pat, slice_pattern);
            walk_list!(visitor, visit_pat, postpatterns);
        }
        PatKind::Mac(ref mac) => visitor.visit_mac(mac),
    }
}

pub fn walk_foreign_item<'a, V: Visitor<'a>>(visitor: &mut V, foreign_item: &'a ForeignItem) {
    visitor.visit_vis(&foreign_item.vis);
    visitor.visit_ident(foreign_item.ident);

    match foreign_item.node {
        ForeignItemKind::Fn(ref function_declaration, ref generics) => {
            walk_fn_decl(visitor, function_declaration);
            visitor.visit_generics(generics)
        }
        ForeignItemKind::Static(ref typ, _) => visitor.visit_ty(typ),
        ForeignItemKind::Ty => (),
        ForeignItemKind::Macro(ref mac) => visitor.visit_mac(mac),
    }

    walk_list!(visitor, visit_attribute, &foreign_item.attrs);
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
        WherePredicate::BoundPredicate(WhereBoundPredicate{ref bounded_ty,
                                                           ref bounds,
                                                           ref bound_generic_params,
                                                           ..}) => {
            visitor.visit_ty(bounded_ty);
            walk_list!(visitor, visit_param_bound, bounds);
            walk_list!(visitor, visit_generic_param, bound_generic_params);
        }
        WherePredicate::RegionPredicate(WhereRegionPredicate{ref lifetime,
                                                             ref bounds,
                                                             ..}) => {
            visitor.visit_lifetime(lifetime);
            walk_list!(visitor, visit_param_bound, bounds);
        }
        WherePredicate::EqPredicate(WhereEqPredicate{ref lhs_ty,
                                                     ref rhs_ty,
                                                     ..}) => {
            visitor.visit_ty(lhs_ty);
            visitor.visit_ty(rhs_ty);
        }
    }
}

pub fn walk_fn_ret_ty<'a, V: Visitor<'a>>(visitor: &mut V, ret_ty: &'a FunctionRetTy) {
    if let FunctionRetTy::Ty(ref output_ty) = *ret_ty {
        visitor.visit_ty(output_ty)
    }
}

pub fn walk_fn_decl<'a, V: Visitor<'a>>(visitor: &mut V, function_declaration: &'a FnDecl) {
    for argument in &function_declaration.inputs {
        walk_list!(visitor, visit_attribute, argument.attrs.iter());
        visitor.visit_pat(&argument.pat);
        visitor.visit_ty(&argument.ty);
    }
    visitor.visit_fn_ret_ty(&function_declaration.output)
}

pub fn walk_fn<'a, V>(visitor: &mut V, kind: FnKind<'a>, declaration: &'a FnDecl, _span: Span)
    where V: Visitor<'a>,
{
    match kind {
        FnKind::ItemFn(_, header, _, body) => {
            visitor.visit_fn_header(header);
            walk_fn_decl(visitor, declaration);
            visitor.visit_block(body);
        }
        FnKind::Method(_, sig, _, body) => {
            visitor.visit_fn_header(&sig.header);
            walk_fn_decl(visitor, declaration);
            visitor.visit_block(body);
        }
        FnKind::Closure(body) => {
            walk_fn_decl(visitor, declaration);
            visitor.visit_expr(body);
        }
    }
}

pub fn walk_trait_item<'a, V: Visitor<'a>>(visitor: &mut V, trait_item: &'a TraitItem) {
    visitor.visit_ident(trait_item.ident);
    walk_list!(visitor, visit_attribute, &trait_item.attrs);
    visitor.visit_generics(&trait_item.generics);
    match trait_item.node {
        TraitItemKind::Const(ref ty, ref default) => {
            visitor.visit_ty(ty);
            walk_list!(visitor, visit_expr, default);
        }
        TraitItemKind::Method(ref sig, None) => {
            visitor.visit_fn_header(&sig.header);
            walk_fn_decl(visitor, &sig.decl);
        }
        TraitItemKind::Method(ref sig, Some(ref body)) => {
            visitor.visit_fn(FnKind::Method(trait_item.ident, sig, None, body),
                             &sig.decl, trait_item.span, trait_item.id);
        }
        TraitItemKind::Type(ref bounds, ref default) => {
            walk_list!(visitor, visit_param_bound, bounds);
            walk_list!(visitor, visit_ty, default);
        }
        TraitItemKind::Macro(ref mac) => {
            visitor.visit_mac(mac);
        }
    }
}

pub fn walk_impl_item<'a, V: Visitor<'a>>(visitor: &mut V, impl_item: &'a ImplItem) {
    visitor.visit_vis(&impl_item.vis);
    visitor.visit_ident(impl_item.ident);
    walk_list!(visitor, visit_attribute, &impl_item.attrs);
    visitor.visit_generics(&impl_item.generics);
    match impl_item.node {
        ImplItemKind::Const(ref ty, ref expr) => {
            visitor.visit_ty(ty);
            visitor.visit_expr(expr);
        }
        ImplItemKind::Method(ref sig, ref body) => {
            visitor.visit_fn(FnKind::Method(impl_item.ident, sig, Some(&impl_item.vis), body),
                             &sig.decl, impl_item.span, impl_item.id);
        }
        ImplItemKind::Type(ref ty) => {
            visitor.visit_ty(ty);
        }
        ImplItemKind::Existential(ref bounds) => {
            walk_list!(visitor, visit_param_bound, bounds);
        }
        ImplItemKind::Macro(ref mac) => {
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
    match statement.node {
        StmtKind::Local(ref local) => visitor.visit_local(local),
        StmtKind::Item(ref item) => visitor.visit_item(item),
        StmtKind::Expr(ref expression) | StmtKind::Semi(ref expression) => {
            visitor.visit_expr(expression)
        }
        StmtKind::Mac(ref mac) => {
            let (ref mac, _, ref attrs) = **mac;
            visitor.visit_mac(mac);
            for attr in attrs.iter() {
                visitor.visit_attribute(attr);
            }
        }
    }
}

pub fn walk_mac<'a, V: Visitor<'a>>(visitor: &mut V, mac: &'a Mac) {
    visitor.visit_path(&mac.node.path, DUMMY_NODE_ID);
}

pub fn walk_anon_const<'a, V: Visitor<'a>>(visitor: &mut V, constant: &'a AnonConst) {
    visitor.visit_expr(&constant.value);
}

pub fn walk_expr<'a, V: Visitor<'a>>(visitor: &mut V, expression: &'a Expr) {
    for attr in expression.attrs.iter() {
        visitor.visit_attribute(attr);
    }
    match expression.node {
        ExprKind::Box(ref subexpression) => {
            visitor.visit_expr(subexpression)
        }
        ExprKind::Array(ref subexpressions) => {
            walk_list!(visitor, visit_expr, subexpressions);
        }
        ExprKind::Repeat(ref element, ref count) => {
            visitor.visit_expr(element);
            visitor.visit_anon_const(count)
        }
        ExprKind::Struct(ref path, ref fields, ref optional_base) => {
            visitor.visit_path(path, expression.id);
            for field in fields {
                walk_list!(visitor, visit_attribute, field.attrs.iter());
                visitor.visit_ident(field.ident);
                visitor.visit_expr(&field.expr)
            }
            walk_list!(visitor, visit_expr, optional_base);
        }
        ExprKind::Tup(ref subexpressions) => {
            walk_list!(visitor, visit_expr, subexpressions);
        }
        ExprKind::Call(ref callee_expression, ref arguments) => {
            visitor.visit_expr(callee_expression);
            walk_list!(visitor, visit_expr, arguments);
        }
        ExprKind::MethodCall(ref segment, ref arguments) => {
            visitor.visit_path_segment(expression.span, segment);
            walk_list!(visitor, visit_expr, arguments);
        }
        ExprKind::Binary(_, ref left_expression, ref right_expression) => {
            visitor.visit_expr(left_expression);
            visitor.visit_expr(right_expression)
        }
        ExprKind::AddrOf(_, ref subexpression) | ExprKind::Unary(_, ref subexpression) => {
            visitor.visit_expr(subexpression)
        }
        ExprKind::Cast(ref subexpression, ref typ) | ExprKind::Type(ref subexpression, ref typ) => {
            visitor.visit_expr(subexpression);
            visitor.visit_ty(typ)
        }
        ExprKind::Let(ref pats, ref scrutinee) => {
            walk_list!(visitor, visit_pat, pats);
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
        ExprKind::Closure(_, _, _, ref function_declaration, ref body, _decl_span) => {
            visitor.visit_fn(FnKind::Closure(body),
                             function_declaration,
                             expression.span,
                             expression.id)
        }
        ExprKind::Block(ref block, ref opt_label) => {
            walk_list!(visitor, visit_label, opt_label);
            visitor.visit_block(block);
        }
        ExprKind::Async(_, _, ref body) => {
            visitor.visit_block(body);
        }
        ExprKind::Await(_, ref expr) => visitor.visit_expr(expr),
        ExprKind::Assign(ref left_hand_expression, ref right_hand_expression) => {
            visitor.visit_expr(left_hand_expression);
            visitor.visit_expr(right_hand_expression);
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
        ExprKind::Mac(ref mac) => visitor.visit_mac(mac),
        ExprKind::Paren(ref subexpression) => {
            visitor.visit_expr(subexpression)
        }
        ExprKind::InlineAsm(ref ia) => {
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
        ExprKind::Try(ref subexpression) => {
            visitor.visit_expr(subexpression)
        }
        ExprKind::TryBlock(ref body) => {
            visitor.visit_block(body)
        }
        ExprKind::Lit(_) | ExprKind::Err => {}
    }

    visitor.visit_expr_post(expression)
}

pub fn walk_arm<'a, V: Visitor<'a>>(visitor: &mut V, arm: &'a Arm) {
    walk_list!(visitor, visit_pat, &arm.pats);
    if let Some(ref e) = &arm.guard {
        visitor.visit_expr(e);
    }
    visitor.visit_expr(&arm.body);
    walk_list!(visitor, visit_attribute, &arm.attrs);
}

pub fn walk_vis<'a, V: Visitor<'a>>(visitor: &mut V, vis: &'a Visibility) {
    if let VisibilityKind::Restricted { ref path, id } = vis.node {
        visitor.visit_path(path, id);
    }
}

pub fn walk_attribute<'a, V: Visitor<'a>>(visitor: &mut V, attr: &'a Attribute) {
    visitor.visit_tts(attr.tokens.clone());
}

pub fn walk_tt<'a, V: Visitor<'a>>(visitor: &mut V, tt: TokenTree) {
    match tt {
        TokenTree::Token(token) => visitor.visit_token(token),
        TokenTree::Delimited(_, _, tts) => visitor.visit_tts(tts),
    }
}

pub fn walk_tts<'a, V: Visitor<'a>>(visitor: &mut V, tts: TokenStream) {
    for tt in tts.trees() {
        visitor.visit_tt(tt);
    }
}
