//! AST walker. Each overridden visit method has full control over what
//! happens with its node, it can do its own traversal of the node's children,
//! call `visit::walk_*` to apply the default traversal algorithm, or prevent
//! deeper traversal by doing nothing.
//!
//! Note: it is an important invariant that the default visitor walks the body
//! of a function in "execution order" (more concretely, reverse post-order
//! with respect to the CFG implied by the AST), meaning that if AST node A may
//! execute before AST node B, then A is visited first.  The borrow checker in
//! particular relies on this property.
//!
//! Note: walking an AST before macro expansion is probably a bad idea. For
//! instance, a walker looking for item names in a module will miss all of
//! those that are created by the expansion of a macro.

use ast::*;
use ptr::P;
use source_map::Spanned;
use syntax_pos::Span;
use parse::token::Token;
use tokenstream::{TokenTree, TokenStream};

pub enum Action<T> {
    Reuse,
    Remove,
    Add(Vec<T>),
    Replace(Vec<T>),
}

impl<T> Action<T> {
    pub fn map<R>(self, mut f: impl FnMut(T) -> R) -> Action<R> {
        match self {
            Action::Reuse => Action::Reuse,
            Action::Remove => Action::Remove,
            Action::Add(list) => {
                Action::Add(list.into_iter().map(|item| f(item)).collect())
            }
            Action::Replace(list) => {
                Action::Replace(list.into_iter().map(|item| f(item)).collect())
            }
        }
    }

    pub fn assert_reuse(self) {
        if let Action::Reuse = self {
        } else {
            panic!()
        }
    }
}

pub enum FnKind<'a> {
    /// fn foo() or extern "Abi" fn foo()
    ItemFn(Ident, FnHeader, &'a mut Visibility, &'a mut Block),

    /// fn foo(&self)
    Method(Ident, FnHeader, Option<&'a mut Visibility>, &'a mut Block),

    /// |x, y| body
    Closure(&'a mut Expr),
}

/// Each method of the MutVisitor trait is a hook to be potentially
/// overridden.  Each method's default implementation recursively visits
/// the substructure of the input via the corresponding `walk` method;
/// e.g., the `visit_mod` method by default calls `visit::walk_mod`.
///
/// If you want to ensure that your code handles every variant
/// explicitly, you need to override each method.  (And you also need
/// to monitor future changes to `MutVisitor` in case a new method with a
/// new default implementation gets introduced.)
pub trait MutVisitor: Sized {
    fn new_id(&mut self, i: NodeId) -> NodeId {
        i
    }

    fn new_span(&mut self, sp: Span) -> Span {
        sp
    }

    fn visit_name(&mut self, _span: Span, _name: Name) {
        // Nothing to do.
    }
    fn visit_ident(&mut self, ident: Ident) {
        walk_ident(self, ident);
    }
    fn visit_crate(&mut self, c: &mut Crate) {
        walk_crate(self, c);
    }
    fn visit_mod(&mut self, m: &mut Mod, _s: Span, _attrs: &[Attribute], _n: NodeId) {
        walk_mod(self, m);
    }
    fn visit_foreign_item(&mut self, i: &mut ForeignItem) -> Action<ForeignItem> {
        walk_foreign_item(self, i);
        Action::Reuse
    }
    fn visit_global_asm(&mut self, ga: &mut GlobalAsm) { walk_global_asm(self, ga) }
    fn visit_item(&mut self, i: &mut P<Item>) -> Action<P<Item>> {
        walk_item(self, i);
        Action::Reuse
    }
    fn visit_local(&mut self, l: &mut Local) { walk_local(self, l) }
    fn visit_block(&mut self, b: &mut Block) { walk_block(self, b) }
    fn visit_stmt(&mut self, s: &mut Stmt) -> Action<Stmt> {
        walk_stmt(self, s)
    }
    fn visit_arm(&mut self, a: &mut Arm) -> Action<Arm> {
        walk_arm(self, a);
        Action::Reuse
    }
    fn visit_field_pat(&mut self, p: &mut Spanned<FieldPat>) -> Action<Spanned<FieldPat>> {
        walk_field_pat(self, p);
        Action::Reuse
    }
    fn visit_pat(&mut self, p: &mut Pat) { walk_pat(self, p) }
    fn visit_anon_const(&mut self, c: &mut AnonConst) { walk_anon_const(self, c) }
    fn visit_field(&mut self, field: &mut Field) -> Action<Field> {
        walk_field(self, field);
        Action::Reuse
    }
    /// Returns true if the expression should be kept
    fn visit_opt_expr(&mut self, ex: &mut Expr) -> bool {
        self.visit_expr(ex);
        true
    }
    fn visit_expr(&mut self, ex: &mut Expr) { walk_expr(self, ex) }
    fn visit_ty(&mut self, t: &mut Ty) { walk_ty(self, t) }
    fn visit_generic_param(&mut self, param: &mut GenericParam) {
        walk_generic_param(self, param)
    }
    fn visit_generics(&mut self, g: &mut Generics) { walk_generics(self, g) }
    fn visit_where_predicate(&mut self, p: &mut WherePredicate) {
        walk_where_predicate(self, p)
    }
    fn visit_fn(&mut self, fk: FnKind<'_>, fd: &mut FnDecl, s: Span, _: NodeId) {
        walk_fn(self, fk, fd, s)
    }
    fn visit_trait_item(&mut self, ti: &mut TraitItem) -> Action<TraitItem> {
        walk_trait_item(self, ti);
        Action::Reuse
    }
    fn visit_impl_item(&mut self, ii: &mut ImplItem) -> Action<ImplItem> {
        walk_impl_item(self, ii);
        Action::Reuse
    }
    fn visit_trait_ref(&mut self, t: &mut TraitRef) { walk_trait_ref(self, t) }
    fn visit_param_bound(&mut self, bounds: &mut GenericBound) {
        walk_param_bound(self, bounds)
    }
    fn visit_poly_trait_ref(&mut self, t: &mut PolyTraitRef, m: &mut TraitBoundModifier) {
        walk_poly_trait_ref(self, t, m)
    }
    fn visit_variant_data(
        &mut self,
        s: &mut VariantData,
        _: Ident,
        _: &mut Generics,
        _: NodeId,
        _: Span
    ) {
        walk_variant_data(self, s)
    }
    fn visit_struct_field(&mut self, s: &mut StructField) -> Action<StructField> {
        walk_struct_field(self, s);
        Action::Reuse
    }
    fn visit_enum_def(&mut self, enum_definition: &mut EnumDef,
                      generics: &mut Generics, item_id: NodeId, _: Span) {
        walk_enum_def(self, enum_definition, generics, item_id)
    }
    fn visit_variant(
        &mut self,
        v: &mut Variant,
        g: &mut Generics,
        item_id: NodeId
    ) -> Action<Variant> {
        walk_variant(self, v, g, item_id);
        Action::Reuse
    }
    fn visit_label(&mut self, label: &mut Label) {
        walk_label(self, label)
    }
    fn visit_lifetime(&mut self, lifetime: &mut Lifetime) {
        walk_lifetime(self, lifetime)
    }
    fn visit_mac(&mut self, _mac: &mut Mac) {
        panic!("visit_mac disabled by default");
        // N.B., see note about macros above.
        // if you really want a visitor that
        // works on macros, use this
        // definition in your trait impl:
        // visit::walk_mac(self, _mac)
    }
    fn visit_mac_def(&mut self, _mac: &mut MacroDef, _id: NodeId) {
        // Nothing to do
    }
    fn visit_path(&mut self, path: &mut Path, _id: NodeId) {
        walk_path(self, path)
    }
    fn visit_use_tree(&mut self, use_tree: &mut UseTree, id: NodeId, _nested: bool) {
        walk_use_tree(self, use_tree, id)
    }
    fn visit_path_segment(&mut self, path_span: Span, path_segment: &mut PathSegment) {
        walk_path_segment(self, path_span, path_segment)
    }
    fn visit_generic_args(&mut self, path_span: Span, generic_args: &mut GenericArgs) {
        walk_generic_args(self, path_span, generic_args)
    }
    fn visit_generic_arg(&mut self, generic_arg: &mut GenericArg) {
        match generic_arg {
            GenericArg::Lifetime(lt) => self.visit_lifetime(lt),
            GenericArg::Type(ty) => self.visit_ty(ty),
        }
    }
    fn visit_assoc_type_binding(&mut self, type_binding: &mut TypeBinding) {
        walk_assoc_type_binding(self, type_binding)
    }
    fn visit_attribute(&mut self, attr: &mut Attribute) {
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
    fn visit_vis(&mut self, vis: &mut Visibility) {
        walk_vis(self, vis)
    }
    fn visit_fn_ret_ty(&mut self, ret_ty: &mut FunctionRetTy) {
        walk_fn_ret_ty(self, ret_ty)
    }
}

#[macro_export]
macro_rules! walk_list_mut {
    ($visitor: expr, $method: ident, $list: expr $(, $extra_args: expr)*) => {
        let mut i = 0;
        loop {
            if i == $list.len() {
                break;
            }

            match $visitor.$method(&mut $list[i] $(, $extra_args)*) {
                Action::Reuse => i += 1,
                Action::Remove => {
                    $list.remove(i);
                }
                Action::Add(list) => {
                    i += 1;
                    let rlen = list.len();
                    for (j, r) in list.into_iter().enumerate() {
                        $list.insert(i + j, r);
                    }
                    i += rlen;
                }
                Action::Replace(list) => {
                    $list.remove(i);
                    let rlen = list.len();
                    for (j, r) in list.into_iter().enumerate() {
                        $list.insert(i + j, r);
                    }
                    i += rlen;
                }
            }
        }
    }
}

pub fn walk_ident<V: MutVisitor>(visitor: &mut V, ident: Ident) {
    visitor.visit_name(ident.span, ident.name);
}

pub fn walk_crate<V: MutVisitor>(visitor: &mut V, krate: &mut Crate) {
    visitor.visit_mod(&mut krate.module, krate.span, &mut krate.attrs, CRATE_NODE_ID);
    walk_list!(visitor, visit_attribute, &mut krate.attrs);
}

pub fn walk_mod<V: MutVisitor>(visitor: &mut V, module: &mut Mod) {
    walk_list_mut!(visitor, visit_item, &mut module.items);
}

pub fn walk_local<V: MutVisitor>(visitor: &mut V, local: &mut Local) {
    for attr in local.attrs.iter_mut() {
        visitor.visit_attribute(attr);
    }
    visitor.visit_pat(&mut local.pat);
    walk_list!(visitor, visit_ty, &mut local.ty);
    walk_list!(visitor, visit_expr, &mut local.init);
}

pub fn walk_label<V: MutVisitor>(visitor: &mut V, label: &mut Label) {
    visitor.visit_ident(label.ident);
}

pub fn walk_lifetime<V: MutVisitor>(visitor: &mut V, lifetime: &mut Lifetime) {
    visitor.visit_ident(lifetime.ident);
}

pub fn walk_poly_trait_ref<V>(visitor: &mut V,
                                  trait_ref: &mut PolyTraitRef,
                                  _: &TraitBoundModifier)
    where V: MutVisitor,
{
    walk_list!(visitor, visit_generic_param, &mut trait_ref.bound_generic_params);
    visitor.visit_trait_ref(&mut trait_ref.trait_ref);
}

pub fn walk_trait_ref<V: MutVisitor>(visitor: &mut V, trait_ref: &mut TraitRef) {
    visitor.visit_path(&mut trait_ref.path, trait_ref.ref_id)
}

pub fn walk_item<V: MutVisitor>(visitor: &mut V, item: &mut Item) {
    visitor.visit_vis(&mut item.vis);
    visitor.visit_ident(item.ident);
    match item.node {
        ItemKind::ExternCrate(orig_name) => {
            if let Some(orig_name) = orig_name {
                visitor.visit_name(item.span, orig_name);
            }
        }
        ItemKind::Use(ref mut use_tree) => {
            visitor.visit_use_tree(use_tree, item.id, false)
        }
        ItemKind::Static(ref mut typ, _, ref mut expr) |
        ItemKind::Const(ref mut typ, ref mut expr) => {
            visitor.visit_ty(typ);
            visitor.visit_expr(expr);
        }
        ItemKind::Fn(ref mut declaration, header, ref mut generics, ref mut body) => {
            visitor.visit_generics(generics);
            visitor.visit_fn(FnKind::ItemFn(item.ident, header,
                                            &mut item.vis, body),
                             declaration,
                             item.span,
                             item.id)
        }
        ItemKind::Mod(ref mut module) => {
            visitor.visit_mod(module, item.span, &mut item.attrs, item.id)
        }
        ItemKind::ForeignMod(ref mut foreign_module) => {
            walk_list_mut!(visitor, visit_foreign_item, &mut foreign_module.items);
        }
        ItemKind::GlobalAsm(ref mut ga) => visitor.visit_global_asm(ga),
        ItemKind::Ty(ref mut typ, ref mut type_parameters) => {
            visitor.visit_ty(typ);
            visitor.visit_generics(type_parameters)
        }
        ItemKind::Existential(ref mut bounds, ref mut type_parameters) => {
            walk_list!(visitor, visit_param_bound, bounds);
            visitor.visit_generics(type_parameters)
        }
        ItemKind::Enum(ref mut enum_definition, ref mut type_parameters) => {
            visitor.visit_generics(type_parameters);
            visitor.visit_enum_def(enum_definition, type_parameters, item.id, item.span)
        }
        ItemKind::Impl(_, _, _,
                 ref mut type_parameters,
                 ref mut opt_trait_reference,
                 ref mut typ,
                 ref mut impl_items) => {
            visitor.visit_generics(type_parameters);
            walk_list!(visitor, visit_trait_ref, opt_trait_reference);
            visitor.visit_ty(typ);
            walk_list_mut!(visitor, visit_impl_item, impl_items);
        }
        ItemKind::Struct(ref mut struct_definition, ref mut generics) |
        ItemKind::Union(ref mut struct_definition, ref mut generics) => {
            visitor.visit_generics(generics);
            visitor.visit_variant_data(struct_definition, item.ident,
                                     generics, item.id, item.span);
        }
        ItemKind::Trait(.., ref mut generics, ref mut bounds, ref mut methods) => {
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_param_bound, bounds);
            walk_list_mut!(visitor, visit_trait_item, methods);
        }
        ItemKind::TraitAlias(ref mut generics, ref mut bounds) => {
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_param_bound, bounds);
        }
        ItemKind::Mac(ref mut mac) => visitor.visit_mac(mac),
        ItemKind::MacroDef(ref mut ts) => visitor.visit_mac_def(ts, item.id),
    }
    walk_list!(visitor, visit_attribute, &mut item.attrs);
}

pub fn walk_enum_def<V: MutVisitor>(visitor: &mut V,
                                 enum_definition: &mut EnumDef,
                                 generics: &mut Generics,
                                 item_id: NodeId) {
    walk_list_mut!(visitor, visit_variant, &mut enum_definition.variants, generics, item_id);
}

pub fn walk_variant<V>(visitor: &mut V,
                           variant: &mut Variant,
                           generics: &mut Generics,
                           item_id: NodeId)
    where V: MutVisitor,
{
    visitor.visit_ident(variant.node.ident);
    visitor.visit_variant_data(&mut variant.node.data, variant.node.ident,
                             generics, item_id, variant.span);
    walk_list!(visitor, visit_anon_const, &mut variant.node.disr_expr);
    walk_list!(visitor, visit_attribute, &mut variant.node.attrs);
}

pub fn walk_ty<V: MutVisitor>(visitor: &mut V, typ: &mut Ty) {
    match typ.node {
        TyKind::Slice(ref mut ty) | TyKind::Paren(ref mut ty) => {
            visitor.visit_ty(ty)
        }
        TyKind::Ptr(ref mut mutable_type) => {
            visitor.visit_ty(&mut mutable_type.ty)
        }
        TyKind::Rptr(ref mut opt_lifetime, ref mut mutable_type) => {
            walk_list!(visitor, visit_lifetime, opt_lifetime);
            visitor.visit_ty(&mut mutable_type.ty)
        }
        TyKind::Never => {},
        TyKind::Tup(ref mut tuple_element_types) => {
            walk_list!(visitor, visit_ty, tuple_element_types);
        }
        TyKind::BareFn(ref mut function_declaration) => {
            walk_list!(visitor, visit_generic_param, &mut function_declaration.generic_params);
            walk_fn_decl(visitor, &mut function_declaration.decl);
        }
        TyKind::Path(ref mut maybe_qself, ref mut path) => {
            if let Some(ref mut qself) = *maybe_qself {
                visitor.visit_ty(&mut qself.ty);
            }
            visitor.visit_path(path, typ.id);
        }
        TyKind::Array(ref mut ty, ref mut length) => {
            visitor.visit_ty(ty);
            visitor.visit_anon_const(length)
        }
        TyKind::TraitObject(ref mut bounds, ..) |
        TyKind::ImplTrait(_, ref mut bounds) => {
            walk_list!(visitor, visit_param_bound, bounds);
        }
        TyKind::Typeof(ref mut expression) => {
            visitor.visit_anon_const(expression)
        }
        TyKind::Infer | TyKind::ImplicitSelf | TyKind::Err => {}
        TyKind::Mac(ref mut mac) => {
            visitor.visit_mac(mac)
        }
    }
}

pub fn walk_path<V: MutVisitor>(visitor: &mut V, path: &mut Path) {
    for segment in &mut path.segments {
        visitor.visit_path_segment(path.span, segment);
    }
}

pub fn walk_use_tree<V: MutVisitor>(
    visitor: &mut V, use_tree: &mut UseTree, id: NodeId,
) {
    visitor.visit_path(&mut use_tree.prefix, id);
    match use_tree.kind {
        UseTreeKind::Simple(rename, ..) => {
            // the extra IDs are handled during HIR lowering
            if let Some(rename) = rename {
                visitor.visit_ident(rename);
            }
        }
        UseTreeKind::Glob => {},
        UseTreeKind::Nested(ref mut use_trees) => {
            for &mut (ref mut nested_tree, nested_id) in use_trees {
                visitor.visit_use_tree(nested_tree, nested_id, true);
            }
        }
    }
}

pub fn walk_path_segment<V: MutVisitor>(visitor: &mut V,
                                             path_span: Span,
                                             segment: &mut PathSegment) {
    visitor.visit_ident(segment.ident);
    if let Some(ref mut args) = segment.args {
        visitor.visit_generic_args(path_span, args);
    }
}

pub fn walk_generic_args<V>(visitor: &mut V,
                                _path_span: Span,
                                generic_args: &mut GenericArgs)
    where V: MutVisitor,
{
    match *generic_args {
        GenericArgs::AngleBracketed(ref mut data) => {
            walk_list!(visitor, visit_generic_arg, &mut data.args);
            walk_list!(visitor, visit_assoc_type_binding, &mut data.bindings);
        }
        GenericArgs::Parenthesized(ref mut data) => {
            walk_list!(visitor, visit_ty, &mut data.inputs);
            walk_list!(visitor, visit_ty, &mut data.output);
        }
    }
}

pub fn walk_assoc_type_binding<V: MutVisitor>(visitor: &mut V,
                                                   type_binding: &mut TypeBinding) {
    visitor.visit_ident(type_binding.ident);
    visitor.visit_ty(&mut type_binding.ty);
}

pub fn walk_field_pat<V: MutVisitor>(visitor: &mut V, field: &mut Spanned<FieldPat>) {
    walk_list!(visitor, visit_attribute, field.node.attrs.iter_mut());
    visitor.visit_ident(field.node.ident);
    visitor.visit_pat(&mut field.node.pat)
}

pub fn walk_pat<V: MutVisitor>(visitor: &mut V, pattern: &mut Pat) {
    match pattern.node {
        PatKind::TupleStruct(ref mut path, ref mut children, _) => {
            visitor.visit_path(path, pattern.id);
            walk_list!(visitor, visit_pat, children);
        }
        PatKind::Path(ref mut opt_qself, ref mut path) => {
            if let Some(ref mut qself) = *opt_qself {
                visitor.visit_ty(&mut qself.ty);
            }
            visitor.visit_path(path, pattern.id)
        }
        PatKind::Struct(ref mut path, ref mut fields, _) => {
            visitor.visit_path(path, pattern.id);
            walk_list_mut!(visitor, visit_field_pat, fields);
        }
        PatKind::Tuple(ref mut tuple_elements, _) => {
            walk_list!(visitor, visit_pat, tuple_elements);
        }
        PatKind::Box(ref mut subpattern) |
        PatKind::Ref(ref mut subpattern, _) |
        PatKind::Paren(ref mut subpattern) => {
            visitor.visit_pat(subpattern)
        }
        PatKind::Ident(_, ident, ref mut optional_subpattern) => {
            visitor.visit_ident(ident);
            walk_list!(visitor, visit_pat, optional_subpattern);
        }
        PatKind::Lit(ref mut expression) => visitor.visit_expr(expression),
        PatKind::Range(ref mut lower_bound, ref mut upper_bound, _) => {
            visitor.visit_expr(lower_bound);
            visitor.visit_expr(upper_bound);
        }
        PatKind::Wild => (),
        PatKind::Slice(ref mut prepatterns, ref mut slice_pattern, ref mut postpatterns) => {
            walk_list!(visitor, visit_pat, prepatterns);
            walk_list!(visitor, visit_pat, slice_pattern);
            walk_list!(visitor, visit_pat, postpatterns);
        }
        PatKind::Mac(ref mut mac) => visitor.visit_mac(mac),
    }
}

pub fn walk_foreign_item<V: MutVisitor>(visitor: &mut V, foreign_item: &mut ForeignItem) {
    visitor.visit_vis(&mut foreign_item.vis);
    visitor.visit_ident(foreign_item.ident);

    match foreign_item.node {
        ForeignItemKind::Fn(ref mut function_declaration, ref mut generics) => {
            walk_fn_decl(visitor, function_declaration);
            visitor.visit_generics(generics)
        }
        ForeignItemKind::Static(ref mut typ, _) => visitor.visit_ty(typ),
        ForeignItemKind::Ty => (),
        ForeignItemKind::Macro(ref mut mac) => visitor.visit_mac(mac),
    }

    walk_list!(visitor, visit_attribute, &mut foreign_item.attrs);
}

pub fn walk_global_asm<V: MutVisitor>(_: &mut V, _: &mut GlobalAsm) {
    // Empty!
}

pub fn walk_param_bound<V: MutVisitor>(visitor: &mut V, bound: &mut GenericBound) {
    match *bound {
        GenericBound::Trait(ref mut typ, ref mut modifier) => {
            visitor.visit_poly_trait_ref(typ, modifier);
        }
        GenericBound::Outlives(ref mut lifetime) => visitor.visit_lifetime(lifetime),
    }
}

pub fn walk_generic_param<V: MutVisitor>(visitor: &mut V, param: &mut GenericParam) {
    visitor.visit_ident(param.ident);
    walk_list!(visitor, visit_attribute, param.attrs.iter_mut());
    walk_list!(visitor, visit_param_bound, &mut param.bounds);
    match param.kind {
        GenericParamKind::Lifetime => {}
        GenericParamKind::Type { ref mut default } => walk_list!(visitor, visit_ty, default),
    }
}

pub fn walk_generics<V: MutVisitor>(visitor: &mut V, generics: &mut Generics) {
    walk_list!(visitor, visit_generic_param, &mut generics.params);
    walk_list!(visitor, visit_where_predicate, &mut generics.where_clause.predicates);
}

pub fn walk_where_predicate<V: MutVisitor>(visitor: &mut V, predicate: &mut WherePredicate) {
    match *predicate {
        WherePredicate::BoundPredicate(WhereBoundPredicate{ref mut bounded_ty,
                                                           ref mut bounds,
                                                           ref mut bound_generic_params,
                                                           ..}) => {
            visitor.visit_ty(bounded_ty);
            walk_list!(visitor, visit_param_bound, bounds);
            walk_list!(visitor, visit_generic_param, bound_generic_params);
        }
        WherePredicate::RegionPredicate(WhereRegionPredicate{ref mut lifetime,
                                                             ref mut bounds,
                                                             ..}) => {
            visitor.visit_lifetime(lifetime);
            walk_list!(visitor, visit_param_bound, bounds);
        }
        WherePredicate::EqPredicate(WhereEqPredicate{ref mut lhs_ty,
                                                     ref mut rhs_ty,
                                                     ..}) => {
            visitor.visit_ty(lhs_ty);
            visitor.visit_ty(rhs_ty);
        }
    }
}

pub fn walk_fn_ret_ty<V: MutVisitor>(visitor: &mut V, ret_ty: &mut FunctionRetTy) {
    if let FunctionRetTy::Ty(ref mut output_ty) = *ret_ty {
        visitor.visit_ty(output_ty)
    }
}

pub fn walk_fn_decl<V: MutVisitor>(visitor: &mut V, function_declaration: &mut FnDecl) {
    for argument in &mut function_declaration.inputs {
        visitor.visit_pat(&mut argument.pat);
        visitor.visit_ty(&mut argument.ty)
    }
    visitor.visit_fn_ret_ty(&mut function_declaration.output)
}

pub fn walk_fn<V>(visitor: &mut V, kind: FnKind<'_>, declaration: &mut FnDecl, _span: Span)
    where V: MutVisitor,
{
    match kind {
        FnKind::ItemFn(_, _, _, body) => {
            walk_fn_decl(visitor, declaration);
            visitor.visit_block(body);
        }
        FnKind::Method(_, _, _, body) => {
            walk_fn_decl(visitor, declaration);
            visitor.visit_block(body);
        }
        FnKind::Closure(body) => {
            walk_fn_decl(visitor, declaration);
            visitor.visit_expr(body);
        }
    }
}

pub fn walk_trait_item<V: MutVisitor>(visitor: &mut V, trait_item: &mut TraitItem) {
    visitor.visit_ident(trait_item.ident);
    walk_list!(visitor, visit_attribute, &mut trait_item.attrs);
    visitor.visit_generics(&mut trait_item.generics);
    match trait_item.node {
        TraitItemKind::Const(ref mut ty, ref mut default) => {
            visitor.visit_ty(ty);
            walk_list!(visitor, visit_expr, default);
        }
        TraitItemKind::Method(ref mut sig, None) => {
            walk_fn_decl(visitor, &mut sig.decl);
        }
        TraitItemKind::Method(ref mut sig, Some(ref mut body)) => {
            visitor.visit_fn(FnKind::Method(trait_item.ident, sig.header, None, body),
                             &mut sig.decl, trait_item.span, trait_item.id);
        }
        TraitItemKind::Type(ref mut bounds, ref mut default) => {
            walk_list!(visitor, visit_param_bound, bounds);
            walk_list!(visitor, visit_ty, default);
        }
        TraitItemKind::Macro(ref mut mac) => {
            visitor.visit_mac(mac);
        }
    }
}

pub fn walk_impl_item<V: MutVisitor>(visitor: &mut V, impl_item: &mut ImplItem) {
    visitor.visit_vis(&mut impl_item.vis);
    visitor.visit_ident(impl_item.ident);
    walk_list!(visitor, visit_attribute, &mut impl_item.attrs);
    visitor.visit_generics(&mut impl_item.generics);
    match impl_item.node {
        ImplItemKind::Const(ref mut ty, ref mut expr) => {
            visitor.visit_ty(ty);
            visitor.visit_expr(expr);
        }
        ImplItemKind::Method(ref mut sig, ref mut body) => {
            visitor.visit_fn(
                FnKind::Method(impl_item.ident, sig.header, Some(&mut impl_item.vis), body),
                &mut sig.decl,
                impl_item.span,
                impl_item.id
            );
        }
        ImplItemKind::Type(ref mut ty) => {
            visitor.visit_ty(ty);
        }
        ImplItemKind::Existential(ref mut bounds) => {
            walk_list!(visitor, visit_param_bound, bounds);
        }
        ImplItemKind::Macro(ref mut mac) => {
            visitor.visit_mac(mac);
        }
    }
}

pub fn walk_variant_data<V: MutVisitor>(visitor: &mut V, variant_data: &mut VariantData) {
    match *variant_data {
        VariantData::Struct(ref mut fields, _) |
        VariantData::Tuple(ref mut fields, _) => {
            walk_list_mut!(visitor, visit_struct_field, fields);
        }
        VariantData::Unit(_) => {}
    }
}

pub fn walk_struct_field<V: MutVisitor>(visitor: &mut V, struct_field: &mut StructField) {
    visitor.visit_vis(&mut struct_field.vis);
    if let Some(ident) = struct_field.ident {
        visitor.visit_ident(ident);
    }
    visitor.visit_ty(&mut struct_field.ty);
    walk_list!(visitor, visit_attribute, &mut struct_field.attrs);
}

pub fn walk_block<V: MutVisitor>(visitor: &mut V, block: &mut Block) {
    walk_list_mut!(visitor, visit_stmt, &mut block.stmts);
}

pub fn walk_stmt<V: MutVisitor>(visitor: &mut V, statement: &mut Stmt) -> Action<Stmt> {
    match statement.node {
        StmtKind::Local(ref mut local) => {
            visitor.visit_local(local);
            Action::Reuse
        },
        StmtKind::Item(ref mut item) => {
            visitor.visit_item(item).map(|item| {
                Stmt {
                    id: visitor.new_id(statement.id),
                    node: StmtKind::Item(item),
                    span: visitor.new_span(statement.span),
                }
            })
        },
        StmtKind::Expr(ref mut expression) | StmtKind::Semi(ref mut expression) => {
            if visitor.visit_opt_expr(expression) {
                Action::Reuse
            } else {
                Action::Remove
            }
        }
        StmtKind::Mac(ref mut mac) => {
            let (ref mut mac, _, ref mut attrs) = **mac;
            visitor.visit_mac(mac);
            for attr in attrs.iter_mut() {
                visitor.visit_attribute(attr);
            }
            Action::Reuse
        }
    }
}

pub fn walk_mac<V: MutVisitor>(_: &mut V, _: &Mac) {
    // Empty!
}

pub fn walk_anon_const<V: MutVisitor>(visitor: &mut V, constant: &mut AnonConst) {
    visitor.visit_expr(&mut constant.value);
}

pub fn walk_field<V: MutVisitor>(visitor: &mut V, field: &mut Field) {
    walk_list!(visitor, visit_attribute, field.attrs.iter_mut());
    visitor.visit_ident(field.ident);
    visitor.visit_expr(&mut field.expr)
}

pub fn walk_expr<V: MutVisitor>(visitor: &mut V, expression: &mut Expr) {
    for attr in expression.attrs.iter_mut() {
        visitor.visit_attribute(attr);
    }
    match expression.node {
        ExprKind::Box(ref mut subexpression) => {
            visitor.visit_expr(subexpression)
        }
        ExprKind::ObsoleteInPlace(ref mut place, ref mut subexpression) => {
            visitor.visit_expr(place);
            visitor.visit_expr(subexpression)
        }
        ExprKind::Array(ref mut subexpressions) => {
            walk_list!(visitor, visit_expr, subexpressions);
        }
        ExprKind::Repeat(ref mut element, ref mut count) => {
            visitor.visit_expr(element);
            visitor.visit_anon_const(count)
        }
        ExprKind::Struct(ref mut path, ref mut fields, ref mut optional_base) => {
            visitor.visit_path(path, expression.id);
            walk_list_mut!(visitor, visit_field, fields);
            walk_list!(visitor, visit_expr, optional_base);
        }
        ExprKind::Tup(ref mut subexpressions) => {
            walk_list!(visitor, visit_expr, subexpressions);
        }
        ExprKind::Call(ref mut callee_expression, ref mut arguments) => {
            visitor.visit_expr(callee_expression);
            walk_list!(visitor, visit_expr, arguments);
        }
        ExprKind::MethodCall(ref mut segment, ref mut arguments) => {
            visitor.visit_path_segment(expression.span, segment);
            walk_list!(visitor, visit_expr, arguments);
        }
        ExprKind::Binary(_, ref mut left_expression, ref mut right_expression) => {
            visitor.visit_expr(left_expression);
            visitor.visit_expr(right_expression)
        }
        ExprKind::AddrOf(_, ref mut subexpression) | ExprKind::Unary(_, ref mut subexpression) => {
            visitor.visit_expr(subexpression)
        }
        ExprKind::Lit(_) => {}
        ExprKind::Cast(ref mut subexpression, ref mut typ) |
        ExprKind::Type(ref mut subexpression, ref mut typ) => {
            visitor.visit_expr(subexpression);
            visitor.visit_ty(typ)
        }
        ExprKind::If(ref mut head_expression, ref mut if_block, ref mut optional_else) => {
            visitor.visit_expr(head_expression);
            visitor.visit_block(if_block);
            walk_list!(visitor, visit_expr, optional_else);
        }
        ExprKind::While(ref mut subexpression, ref mut block, ref mut opt_label) => {
            walk_list!(visitor, visit_label, opt_label);
            visitor.visit_expr(subexpression);
            visitor.visit_block(block);
        }
        ExprKind::IfLet(
            ref mut pats,
            ref mut subexpression,
            ref mut if_block,
            ref mut optional_else
        ) => {
            walk_list!(visitor, visit_pat, pats);
            visitor.visit_expr(subexpression);
            visitor.visit_block(if_block);
            walk_list!(visitor, visit_expr, optional_else);
        }
        ExprKind::WhileLet(
            ref mut pats,
            ref mut subexpression,
            ref mut block,
            ref mut opt_label
        ) => {
            walk_list!(visitor, visit_label, opt_label);
            walk_list!(visitor, visit_pat, pats);
            visitor.visit_expr(subexpression);
            visitor.visit_block(block);
        }
        ExprKind::ForLoop(
            ref mut pattern,
            ref mut subexpression,
            ref mut block,
            ref mut opt_label
        ) => {
            walk_list!(visitor, visit_label, opt_label);
            visitor.visit_pat(pattern);
            visitor.visit_expr(subexpression);
            visitor.visit_block(block);
        }
        ExprKind::Loop(ref mut block, ref mut opt_label) => {
            walk_list!(visitor, visit_label, opt_label);
            visitor.visit_block(block);
        }
        ExprKind::Match(ref mut subexpression, ref mut arms) => {
            visitor.visit_expr(subexpression);
            walk_list_mut!(visitor, visit_arm, arms);
        }
        ExprKind::Closure(_, _, _, ref mut function_declaration, ref mut body, _decl_span) => {
            visitor.visit_fn(FnKind::Closure(body),
                             function_declaration,
                             expression.span,
                             expression.id)
        }
        ExprKind::Block(ref mut block, ref mut opt_label) => {
            walk_list!(visitor, visit_label, opt_label);
            visitor.visit_block(block);
        }
        ExprKind::Async(_, _, ref mut body) => {
            visitor.visit_block(body);
        }
        ExprKind::Assign(ref mut left_hand_expression, ref mut right_hand_expression) => {
            visitor.visit_expr(left_hand_expression);
            visitor.visit_expr(right_hand_expression);
        }
        ExprKind::AssignOp(_, ref mut left_expression, ref mut right_expression) => {
            visitor.visit_expr(left_expression);
            visitor.visit_expr(right_expression);
        }
        ExprKind::Field(ref mut subexpression, ident) => {
            visitor.visit_expr(subexpression);
            visitor.visit_ident(ident);
        }
        ExprKind::Index(ref mut main_expression, ref mut index_expression) => {
            visitor.visit_expr(main_expression);
            visitor.visit_expr(index_expression)
        }
        ExprKind::Range(ref mut start, ref mut end, _) => {
            walk_list!(visitor, visit_expr, start);
            walk_list!(visitor, visit_expr, end);
        }
        ExprKind::Path(ref mut maybe_qself, ref mut path) => {
            if let Some(ref mut qself) = *maybe_qself {
                visitor.visit_ty(&mut qself.ty);
            }
            visitor.visit_path(path, expression.id)
        }
        ExprKind::Break(ref mut opt_label, ref mut opt_expr) => {
            walk_list!(visitor, visit_label, opt_label);
            walk_list!(visitor, visit_expr, opt_expr);
        }
        ExprKind::Continue(ref mut opt_label) => {
            walk_list!(visitor, visit_label, opt_label);
        }
        ExprKind::Ret(ref mut optional_expression) => {
            walk_list!(visitor, visit_expr, optional_expression);
        }
        ExprKind::Mac(ref mut mac) => visitor.visit_mac(mac),
        ExprKind::Paren(ref mut subexpression) => {
            visitor.visit_expr(subexpression)
        }
        ExprKind::InlineAsm(ref mut ia) => {
            for &mut (_, ref mut input) in &mut ia.inputs {
                visitor.visit_expr(input)
            }
            for output in &mut ia.outputs {
                visitor.visit_expr(&mut output.expr)
            }
        }
        ExprKind::Yield(ref mut optional_expression) => {
            walk_list!(visitor, visit_expr, optional_expression);
        }
        ExprKind::Try(ref mut subexpression) => {
            visitor.visit_expr(subexpression)
        }
        ExprKind::TryBlock(ref mut body) => {
            visitor.visit_block(body)
        }
        ExprKind::Err => {}
    }
}

pub fn walk_arm<V: MutVisitor>(visitor: &mut V, arm: &mut Arm) {
    walk_list!(visitor, visit_pat, &mut arm.pats);
    if let Some(ref mut g) = &mut arm.guard {
        match g {
            Guard::If(ref mut e) => visitor.visit_expr(e),
        }
    }
    visitor.visit_expr(&mut arm.body);
    walk_list!(visitor, visit_attribute, &mut arm.attrs);
}

pub fn walk_vis<V: MutVisitor>(visitor: &mut V, vis: &mut Visibility) {
    if let VisibilityKind::Restricted { ref mut path, id } = vis.node {
        visitor.visit_path(path, id);
    }
}

pub fn walk_attribute<V: MutVisitor>(visitor: &mut V, attr: &mut Attribute) {
    visitor.visit_tts(attr.tokens.clone());
}

pub fn walk_tt<V: MutVisitor>(visitor: &mut V, tt: TokenTree) {
    match tt {
        TokenTree::Token(_, tok) => visitor.visit_token(tok),
        TokenTree::Delimited(_, _, tts) => visitor.visit_tts(tts),
    }
}

pub fn walk_tts<V: MutVisitor>(visitor: &mut V, tts: TokenStream) {
    for tt in tts.trees() {
        visitor.visit_tt(tt);
    }
}
