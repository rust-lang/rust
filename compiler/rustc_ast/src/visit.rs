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

use rustc_span::symbol::Ident;
use rustc_span::Span;

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum AssocCtxt {
    Trait,
    Impl,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum FnCtxt {
    Free,
    Foreign,
    Assoc(AssocCtxt),
}

#[derive(Copy, Clone, Debug)]
pub enum BoundKind {
    /// Trait bounds in generics bounds and type/trait alias.
    /// E.g., `<T: Bound>`, `type A: Bound`, or `where T: Bound`.
    Bound,

    /// Trait bounds in `impl` type.
    /// E.g., `type Foo = impl Bound1 + Bound2 + Bound3`.
    Impl,

    /// Trait bounds in trait object type.
    /// E.g., `dyn Bound1 + Bound2 + Bound3`.
    TraitObject,

    /// Super traits of a trait.
    /// E.g., `trait A: B`
    SuperTraits,
}

#[derive(Copy, Clone, Debug)]
pub enum FnKind<'a> {
    /// E.g., `fn foo()`, `fn foo(&self)`, or `extern "Abi" fn foo()`.
    Fn(FnCtxt, Ident, &'a FnSig, &'a Visibility, &'a Generics, Option<&'a Block>),

    /// E.g., `|x, y| body`.
    Closure(&'a ClosureBinder, &'a FnDecl, &'a Expr),
}

impl<'a> FnKind<'a> {
    pub fn header(&self) -> Option<&'a FnHeader> {
        match *self {
            FnKind::Fn(_, _, sig, _, _, _) => Some(&sig.header),
            FnKind::Closure(_, _, _) => None,
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
            FnKind::Fn(_, _, sig, _, _, _) => &sig.decl,
            FnKind::Closure(_, decl, _) => decl,
        }
    }

    pub fn ctxt(&self) -> Option<FnCtxt> {
        match self {
            FnKind::Fn(ctxt, ..) => Some(*ctxt),
            FnKind::Closure(..) => None,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum LifetimeCtxt {
    /// Appears in a reference type.
    Ref,
    /// Appears as a bound on a type or another lifetime.
    Bound,
    /// Appears as a generic argument.
    GenericArg,
}

/// Each method of the `Visitor` trait is a hook to be potentially
/// overridden. Each method's default implementation recursively visits
/// the substructure of the input via the corresponding `walk` method;
/// e.g., the `visit_item` method by default calls `visit::walk_item`.
///
/// If you want to ensure that your code handles every variant
/// explicitly, you need to override each method. (And you also need
/// to monitor future changes to `Visitor` in case a new method with a
/// new default implementation gets introduced.)
pub trait Visitor<'ast>: Sized {
    fn visit_ident(&mut self, _ident: Ident) {}
    fn visit_foreign_item(&mut self, i: &'ast ForeignItem) {
        walk_foreign_item(self, i)
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
    /// This method is a hack to workaround unstable of `stmt_expr_attributes`.
    /// It can be removed once that feature is stabilized.
    fn visit_method_receiver_expr(&mut self, ex: &'ast Expr) {
        self.visit_expr(ex)
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
    fn visit_closure_binder(&mut self, b: &'ast ClosureBinder) {
        walk_closure_binder(self, b)
    }
    fn visit_where_predicate(&mut self, p: &'ast WherePredicate) {
        walk_where_predicate(self, p)
    }
    fn visit_fn(&mut self, fk: FnKind<'ast>, _: Span, _: NodeId) {
        walk_fn(self, fk)
    }
    fn visit_assoc_item(&mut self, i: &'ast AssocItem, ctxt: AssocCtxt) {
        walk_assoc_item(self, i, ctxt)
    }
    fn visit_trait_ref(&mut self, t: &'ast TraitRef) {
        walk_trait_ref(self, t)
    }
    fn visit_param_bound(&mut self, bounds: &'ast GenericBound, _ctxt: BoundKind) {
        walk_param_bound(self, bounds)
    }
    fn visit_poly_trait_ref(&mut self, t: &'ast PolyTraitRef) {
        walk_poly_trait_ref(self, t)
    }
    fn visit_variant_data(&mut self, s: &'ast VariantData) {
        walk_struct_def(self, s)
    }
    fn visit_field_def(&mut self, s: &'ast FieldDef) {
        walk_field_def(self, s)
    }
    fn visit_enum_def(&mut self, enum_definition: &'ast EnumDef) {
        walk_enum_def(self, enum_definition)
    }
    fn visit_variant(&mut self, v: &'ast Variant) {
        walk_variant(self, v)
    }
    fn visit_label(&mut self, label: &'ast Label) {
        walk_label(self, label)
    }
    fn visit_lifetime(&mut self, lifetime: &'ast Lifetime, _: LifetimeCtxt) {
        walk_lifetime(self, lifetime)
    }
    fn visit_mac_call(&mut self, mac: &'ast MacCall) {
        walk_mac(self, mac)
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
    fn visit_path_segment(&mut self, path_segment: &'ast PathSegment) {
        walk_path_segment(self, path_segment)
    }
    fn visit_generic_args(&mut self, generic_args: &'ast GenericArgs) {
        walk_generic_args(self, generic_args)
    }
    fn visit_generic_arg(&mut self, generic_arg: &'ast GenericArg) {
        walk_generic_arg(self, generic_arg)
    }
    fn visit_assoc_constraint(&mut self, constraint: &'ast AssocConstraint) {
        walk_assoc_constraint(self, constraint)
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
    fn visit_expr_field(&mut self, f: &'ast ExprField) {
        walk_expr_field(self, f)
    }
    fn visit_pat_field(&mut self, fp: &'ast PatField) {
        walk_pat_field(self, fp)
    }
    fn visit_crate(&mut self, krate: &'ast Crate) {
        walk_crate(self, krate)
    }
    fn visit_inline_asm(&mut self, asm: &'ast InlineAsm) {
        walk_inline_asm(self, asm)
    }
    fn visit_format_args(&mut self, fmt: &'ast FormatArgs) {
        walk_format_args(self, fmt)
    }
    fn visit_inline_asm_sym(&mut self, sym: &'ast InlineAsmSym) {
        walk_inline_asm_sym(self, sym)
    }
}

#[macro_export]
macro_rules! walk_list {
    ($visitor: expr, $method: ident, $list: expr $(, $($extra_args: expr),* )?) => {
        {
            #[allow(for_loops_over_fallibles)]
            for elem in $list {
                $visitor.$method(elem $(, $($extra_args,)* )?)
            }
        }
    }
}

pub fn walk_crate<'a, V: Visitor<'a>>(visitor: &mut V, krate: &'a Crate) {
    walk_list!(visitor, visit_item, &krate.items);
    walk_list!(visitor, visit_attribute, &krate.attrs);
}

pub fn walk_local<'a, V: Visitor<'a>>(visitor: &mut V, local: &'a Local) {
    for attr in local.attrs.iter() {
        visitor.visit_attribute(attr);
    }
    visitor.visit_pat(&local.pat);
    walk_list!(visitor, visit_ty, &local.ty);
    if let Some((init, els)) = local.kind.init_else_opt() {
        visitor.visit_expr(init);
        walk_list!(visitor, visit_block, els);
    }
}

pub fn walk_label<'a, V: Visitor<'a>>(visitor: &mut V, label: &'a Label) {
    visitor.visit_ident(label.ident);
}

pub fn walk_lifetime<'a, V: Visitor<'a>>(visitor: &mut V, lifetime: &'a Lifetime) {
    visitor.visit_ident(lifetime.ident);
}

pub fn walk_poly_trait_ref<'a, V>(visitor: &mut V, trait_ref: &'a PolyTraitRef)
where
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
    match &item.kind {
        ItemKind::ExternCrate(_) => {}
        ItemKind::Use(use_tree) => visitor.visit_use_tree(use_tree, item.id, false),
        ItemKind::Static(typ, _, expr) | ItemKind::Const(_, typ, expr) => {
            visitor.visit_ty(typ);
            walk_list!(visitor, visit_expr, expr);
        }
        ItemKind::Fn(box Fn { defaultness: _, generics, sig, body }) => {
            let kind =
                FnKind::Fn(FnCtxt::Free, item.ident, sig, &item.vis, generics, body.as_deref());
            visitor.visit_fn(kind, item.span, item.id)
        }
        ItemKind::Mod(_unsafety, mod_kind) => match mod_kind {
            ModKind::Loaded(items, _inline, _inner_span) => {
                walk_list!(visitor, visit_item, items)
            }
            ModKind::Unloaded => {}
        },
        ItemKind::ForeignMod(foreign_module) => {
            walk_list!(visitor, visit_foreign_item, &foreign_module.items);
        }
        ItemKind::GlobalAsm(asm) => visitor.visit_inline_asm(asm),
        ItemKind::TyAlias(box TyAlias { generics, bounds, ty, .. }) => {
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
            walk_list!(visitor, visit_ty, ty);
        }
        ItemKind::Enum(enum_definition, generics) => {
            visitor.visit_generics(generics);
            visitor.visit_enum_def(enum_definition)
        }
        ItemKind::Impl(box Impl {
            defaultness: _,
            unsafety: _,
            generics,
            constness: _,
            polarity: _,
            of_trait,
            self_ty,
            items,
        }) => {
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_trait_ref, of_trait);
            visitor.visit_ty(self_ty);
            walk_list!(visitor, visit_assoc_item, items, AssocCtxt::Impl);
        }
        ItemKind::Struct(struct_definition, generics)
        | ItemKind::Union(struct_definition, generics) => {
            visitor.visit_generics(generics);
            visitor.visit_variant_data(struct_definition);
        }
        ItemKind::Trait(box Trait { unsafety: _, is_auto: _, generics, bounds, items }) => {
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::SuperTraits);
            walk_list!(visitor, visit_assoc_item, items, AssocCtxt::Trait);
        }
        ItemKind::TraitAlias(generics, bounds) => {
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
        }
        ItemKind::MacCall(mac) => visitor.visit_mac_call(mac),
        ItemKind::MacroDef(ts) => visitor.visit_mac_def(ts, item.id),
    }
    walk_list!(visitor, visit_attribute, &item.attrs);
}

pub fn walk_enum_def<'a, V: Visitor<'a>>(visitor: &mut V, enum_definition: &'a EnumDef) {
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

pub fn walk_expr_field<'a, V: Visitor<'a>>(visitor: &mut V, f: &'a ExprField) {
    visitor.visit_expr(&f.expr);
    visitor.visit_ident(f.ident);
    walk_list!(visitor, visit_attribute, f.attrs.iter());
}

pub fn walk_pat_field<'a, V: Visitor<'a>>(visitor: &mut V, fp: &'a PatField) {
    visitor.visit_ident(fp.ident);
    visitor.visit_pat(&fp.pat);
    walk_list!(visitor, visit_attribute, fp.attrs.iter());
}

pub fn walk_ty<'a, V: Visitor<'a>>(visitor: &mut V, typ: &'a Ty) {
    match &typ.kind {
        TyKind::Slice(ty) | TyKind::Paren(ty) => visitor.visit_ty(ty),
        TyKind::Ptr(mutable_type) => visitor.visit_ty(&mutable_type.ty),
        TyKind::Ref(opt_lifetime, mutable_type) => {
            walk_list!(visitor, visit_lifetime, opt_lifetime, LifetimeCtxt::Ref);
            visitor.visit_ty(&mutable_type.ty)
        }
        TyKind::Tup(tuple_element_types) => {
            walk_list!(visitor, visit_ty, tuple_element_types);
        }
        TyKind::BareFn(function_declaration) => {
            walk_list!(visitor, visit_generic_param, &function_declaration.generic_params);
            walk_fn_decl(visitor, &function_declaration.decl);
        }
        TyKind::Path(maybe_qself, path) => {
            if let Some(qself) = maybe_qself {
                visitor.visit_ty(&qself.ty);
            }
            visitor.visit_path(path, typ.id);
        }
        TyKind::Array(ty, length) => {
            visitor.visit_ty(ty);
            visitor.visit_anon_const(length)
        }
        TyKind::TraitObject(bounds, ..) => {
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::TraitObject);
        }
        TyKind::ImplTrait(_, bounds) => {
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::Impl);
        }
        TyKind::Typeof(expression) => visitor.visit_anon_const(expression),
        TyKind::Infer | TyKind::ImplicitSelf | TyKind::Err => {}
        TyKind::MacCall(mac) => visitor.visit_mac_call(mac),
        TyKind::Never | TyKind::CVarArgs => {}
    }
}

pub fn walk_path<'a, V: Visitor<'a>>(visitor: &mut V, path: &'a Path) {
    for segment in &path.segments {
        visitor.visit_path_segment(segment);
    }
}

pub fn walk_use_tree<'a, V: Visitor<'a>>(visitor: &mut V, use_tree: &'a UseTree, id: NodeId) {
    visitor.visit_path(&use_tree.prefix, id);
    match &use_tree.kind {
        UseTreeKind::Simple(rename) => {
            // The extra IDs are handled during HIR lowering.
            if let &Some(rename) = rename {
                visitor.visit_ident(rename);
            }
        }
        UseTreeKind::Glob => {}
        UseTreeKind::Nested(use_trees) => {
            for &(ref nested_tree, nested_id) in use_trees {
                visitor.visit_use_tree(nested_tree, nested_id, true);
            }
        }
    }
}

pub fn walk_path_segment<'a, V: Visitor<'a>>(visitor: &mut V, segment: &'a PathSegment) {
    visitor.visit_ident(segment.ident);
    if let Some(args) = &segment.args {
        visitor.visit_generic_args(args);
    }
}

pub fn walk_generic_args<'a, V>(visitor: &mut V, generic_args: &'a GenericArgs)
where
    V: Visitor<'a>,
{
    match generic_args {
        GenericArgs::AngleBracketed(data) => {
            for arg in &data.args {
                match arg {
                    AngleBracketedArg::Arg(a) => visitor.visit_generic_arg(a),
                    AngleBracketedArg::Constraint(c) => visitor.visit_assoc_constraint(c),
                }
            }
        }
        GenericArgs::Parenthesized(data) => {
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
        GenericArg::Lifetime(lt) => visitor.visit_lifetime(lt, LifetimeCtxt::GenericArg),
        GenericArg::Type(ty) => visitor.visit_ty(ty),
        GenericArg::Const(ct) => visitor.visit_anon_const(ct),
    }
}

pub fn walk_assoc_constraint<'a, V: Visitor<'a>>(visitor: &mut V, constraint: &'a AssocConstraint) {
    visitor.visit_ident(constraint.ident);
    if let Some(gen_args) = &constraint.gen_args {
        visitor.visit_generic_args(gen_args);
    }
    match &constraint.kind {
        AssocConstraintKind::Equality { term } => match term {
            Term::Ty(ty) => visitor.visit_ty(ty),
            Term::Const(c) => visitor.visit_anon_const(c),
        },
        AssocConstraintKind::Bound { bounds } => {
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
        }
    }
}

pub fn walk_pat<'a, V: Visitor<'a>>(visitor: &mut V, pattern: &'a Pat) {
    match &pattern.kind {
        PatKind::TupleStruct(opt_qself, path, elems) => {
            if let Some(qself) = opt_qself {
                visitor.visit_ty(&qself.ty);
            }
            visitor.visit_path(path, pattern.id);
            walk_list!(visitor, visit_pat, elems);
        }
        PatKind::Path(opt_qself, path) => {
            if let Some(qself) = opt_qself {
                visitor.visit_ty(&qself.ty);
            }
            visitor.visit_path(path, pattern.id)
        }
        PatKind::Struct(opt_qself, path, fields, _) => {
            if let Some(qself) = opt_qself {
                visitor.visit_ty(&qself.ty);
            }
            visitor.visit_path(path, pattern.id);
            walk_list!(visitor, visit_pat_field, fields);
        }
        PatKind::Box(subpattern) | PatKind::Ref(subpattern, _) | PatKind::Paren(subpattern) => {
            visitor.visit_pat(subpattern)
        }
        PatKind::Ident(_, ident, optional_subpattern) => {
            visitor.visit_ident(*ident);
            walk_list!(visitor, visit_pat, optional_subpattern);
        }
        PatKind::Lit(expression) => visitor.visit_expr(expression),
        PatKind::Range(lower_bound, upper_bound, _) => {
            walk_list!(visitor, visit_expr, lower_bound);
            walk_list!(visitor, visit_expr, upper_bound);
        }
        PatKind::Wild | PatKind::Rest => {}
        PatKind::Tuple(elems) | PatKind::Slice(elems) | PatKind::Or(elems) => {
            walk_list!(visitor, visit_pat, elems);
        }
        PatKind::MacCall(mac) => visitor.visit_mac_call(mac),
    }
}

pub fn walk_foreign_item<'a, V: Visitor<'a>>(visitor: &mut V, item: &'a ForeignItem) {
    let &Item { id, span, ident, ref vis, ref attrs, ref kind, tokens: _ } = item;
    visitor.visit_vis(vis);
    visitor.visit_ident(ident);
    walk_list!(visitor, visit_attribute, attrs);
    match kind {
        ForeignItemKind::Static(ty, _, expr) => {
            visitor.visit_ty(ty);
            walk_list!(visitor, visit_expr, expr);
        }
        ForeignItemKind::Fn(box Fn { defaultness: _, generics, sig, body }) => {
            let kind = FnKind::Fn(FnCtxt::Foreign, ident, sig, vis, generics, body.as_deref());
            visitor.visit_fn(kind, span, id);
        }
        ForeignItemKind::TyAlias(box TyAlias { generics, bounds, ty, .. }) => {
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
            walk_list!(visitor, visit_ty, ty);
        }
        ForeignItemKind::MacCall(mac) => {
            visitor.visit_mac_call(mac);
        }
    }
}

pub fn walk_param_bound<'a, V: Visitor<'a>>(visitor: &mut V, bound: &'a GenericBound) {
    match bound {
        GenericBound::Trait(typ, _modifier) => visitor.visit_poly_trait_ref(typ),
        GenericBound::Outlives(lifetime) => visitor.visit_lifetime(lifetime, LifetimeCtxt::Bound),
    }
}

pub fn walk_generic_param<'a, V: Visitor<'a>>(visitor: &mut V, param: &'a GenericParam) {
    visitor.visit_ident(param.ident);
    walk_list!(visitor, visit_attribute, param.attrs.iter());
    walk_list!(visitor, visit_param_bound, &param.bounds, BoundKind::Bound);
    match &param.kind {
        GenericParamKind::Lifetime => (),
        GenericParamKind::Type { default } => walk_list!(visitor, visit_ty, default),
        GenericParamKind::Const { ty, default, .. } => {
            visitor.visit_ty(ty);
            if let Some(default) = default {
                visitor.visit_anon_const(default);
            }
        }
    }
}

pub fn walk_generics<'a, V: Visitor<'a>>(visitor: &mut V, generics: &'a Generics) {
    walk_list!(visitor, visit_generic_param, &generics.params);
    walk_list!(visitor, visit_where_predicate, &generics.where_clause.predicates);
}

pub fn walk_closure_binder<'a, V: Visitor<'a>>(visitor: &mut V, binder: &'a ClosureBinder) {
    match binder {
        ClosureBinder::NotPresent => {}
        ClosureBinder::For { generic_params, span: _ } => {
            walk_list!(visitor, visit_generic_param, generic_params)
        }
    }
}

pub fn walk_where_predicate<'a, V: Visitor<'a>>(visitor: &mut V, predicate: &'a WherePredicate) {
    match predicate {
        WherePredicate::BoundPredicate(WhereBoundPredicate {
            bounded_ty,
            bounds,
            bound_generic_params,
            ..
        }) => {
            visitor.visit_ty(bounded_ty);
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
            walk_list!(visitor, visit_generic_param, bound_generic_params);
        }
        WherePredicate::RegionPredicate(WhereRegionPredicate { lifetime, bounds, .. }) => {
            visitor.visit_lifetime(lifetime, LifetimeCtxt::Bound);
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
        }
        WherePredicate::EqPredicate(WhereEqPredicate { lhs_ty, rhs_ty, .. }) => {
            visitor.visit_ty(lhs_ty);
            visitor.visit_ty(rhs_ty);
        }
    }
}

pub fn walk_fn_ret_ty<'a, V: Visitor<'a>>(visitor: &mut V, ret_ty: &'a FnRetTy) {
    if let FnRetTy::Ty(output_ty) = ret_ty {
        visitor.visit_ty(output_ty)
    }
}

pub fn walk_fn_decl<'a, V: Visitor<'a>>(visitor: &mut V, function_declaration: &'a FnDecl) {
    for param in &function_declaration.inputs {
        visitor.visit_param(param);
    }
    visitor.visit_fn_ret_ty(&function_declaration.output);
}

pub fn walk_fn<'a, V: Visitor<'a>>(visitor: &mut V, kind: FnKind<'a>) {
    match kind {
        FnKind::Fn(_, _, sig, _, generics, body) => {
            visitor.visit_generics(generics);
            visitor.visit_fn_header(&sig.header);
            walk_fn_decl(visitor, &sig.decl);
            walk_list!(visitor, visit_block, body);
        }
        FnKind::Closure(binder, decl, body) => {
            visitor.visit_closure_binder(binder);
            walk_fn_decl(visitor, decl);
            visitor.visit_expr(body);
        }
    }
}

pub fn walk_assoc_item<'a, V: Visitor<'a>>(visitor: &mut V, item: &'a AssocItem, ctxt: AssocCtxt) {
    let &Item { id, span, ident, ref vis, ref attrs, ref kind, tokens: _ } = item;
    visitor.visit_vis(vis);
    visitor.visit_ident(ident);
    walk_list!(visitor, visit_attribute, attrs);
    match kind {
        AssocItemKind::Const(_, ty, expr) => {
            visitor.visit_ty(ty);
            walk_list!(visitor, visit_expr, expr);
        }
        AssocItemKind::Fn(box Fn { defaultness: _, generics, sig, body }) => {
            let kind = FnKind::Fn(FnCtxt::Assoc(ctxt), ident, sig, vis, generics, body.as_deref());
            visitor.visit_fn(kind, span, id);
        }
        AssocItemKind::Type(box TyAlias { generics, bounds, ty, .. }) => {
            visitor.visit_generics(generics);
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
            walk_list!(visitor, visit_ty, ty);
        }
        AssocItemKind::MacCall(mac) => {
            visitor.visit_mac_call(mac);
        }
    }
}

pub fn walk_struct_def<'a, V: Visitor<'a>>(visitor: &mut V, struct_definition: &'a VariantData) {
    walk_list!(visitor, visit_field_def, struct_definition.fields());
}

pub fn walk_field_def<'a, V: Visitor<'a>>(visitor: &mut V, field: &'a FieldDef) {
    visitor.visit_vis(&field.vis);
    if let Some(ident) = field.ident {
        visitor.visit_ident(ident);
    }
    visitor.visit_ty(&field.ty);
    walk_list!(visitor, visit_attribute, &field.attrs);
}

pub fn walk_block<'a, V: Visitor<'a>>(visitor: &mut V, block: &'a Block) {
    walk_list!(visitor, visit_stmt, &block.stmts);
}

pub fn walk_stmt<'a, V: Visitor<'a>>(visitor: &mut V, statement: &'a Stmt) {
    match &statement.kind {
        StmtKind::Local(local) => visitor.visit_local(local),
        StmtKind::Item(item) => visitor.visit_item(item),
        StmtKind::Expr(expr) | StmtKind::Semi(expr) => visitor.visit_expr(expr),
        StmtKind::Empty => {}
        StmtKind::MacCall(mac) => {
            let MacCallStmt { mac, attrs, style: _, tokens: _ } = &**mac;
            visitor.visit_mac_call(mac);
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

pub fn walk_inline_asm<'a, V: Visitor<'a>>(visitor: &mut V, asm: &'a InlineAsm) {
    for (op, _) in &asm.operands {
        match op {
            InlineAsmOperand::In { expr, .. }
            | InlineAsmOperand::Out { expr: Some(expr), .. }
            | InlineAsmOperand::InOut { expr, .. } => visitor.visit_expr(expr),
            InlineAsmOperand::Out { expr: None, .. } => {}
            InlineAsmOperand::SplitInOut { in_expr, out_expr, .. } => {
                visitor.visit_expr(in_expr);
                if let Some(out_expr) = out_expr {
                    visitor.visit_expr(out_expr);
                }
            }
            InlineAsmOperand::Const { anon_const, .. } => visitor.visit_anon_const(anon_const),
            InlineAsmOperand::Sym { sym } => visitor.visit_inline_asm_sym(sym),
        }
    }
}

pub fn walk_inline_asm_sym<'a, V: Visitor<'a>>(visitor: &mut V, sym: &'a InlineAsmSym) {
    if let Some(qself) = &sym.qself {
        visitor.visit_ty(&qself.ty);
    }
    visitor.visit_path(&sym.path, sym.id);
}

pub fn walk_format_args<'a, V: Visitor<'a>>(visitor: &mut V, fmt: &'a FormatArgs) {
    for arg in fmt.arguments.all_args() {
        if let FormatArgumentKind::Named(name) = arg.kind {
            visitor.visit_ident(name);
        }
        visitor.visit_expr(&arg.expr);
    }
}

pub fn walk_expr<'a, V: Visitor<'a>>(visitor: &mut V, expression: &'a Expr) {
    walk_list!(visitor, visit_attribute, expression.attrs.iter());

    match &expression.kind {
        ExprKind::Array(subexpressions) => {
            walk_list!(visitor, visit_expr, subexpressions);
        }
        ExprKind::ConstBlock(anon_const) => visitor.visit_anon_const(anon_const),
        ExprKind::Repeat(element, count) => {
            visitor.visit_expr(element);
            visitor.visit_anon_const(count)
        }
        ExprKind::Struct(se) => {
            if let Some(qself) = &se.qself {
                visitor.visit_ty(&qself.ty);
            }
            visitor.visit_path(&se.path, expression.id);
            walk_list!(visitor, visit_expr_field, &se.fields);
            match &se.rest {
                StructRest::Base(expr) => visitor.visit_expr(expr),
                StructRest::Rest(_span) => {}
                StructRest::None => {}
            }
        }
        ExprKind::Tup(subexpressions) => {
            walk_list!(visitor, visit_expr, subexpressions);
        }
        ExprKind::Call(callee_expression, arguments) => {
            visitor.visit_expr(callee_expression);
            walk_list!(visitor, visit_expr, arguments);
        }
        ExprKind::MethodCall(box MethodCall { seg, receiver, args, span: _ }) => {
            visitor.visit_path_segment(seg);
            visitor.visit_expr(receiver);
            walk_list!(visitor, visit_expr, args);
        }
        ExprKind::Binary(_, left_expression, right_expression) => {
            visitor.visit_expr(left_expression);
            visitor.visit_expr(right_expression)
        }
        ExprKind::AddrOf(_, _, subexpression) | ExprKind::Unary(_, subexpression) => {
            visitor.visit_expr(subexpression)
        }
        ExprKind::Cast(subexpression, typ) | ExprKind::Type(subexpression, typ) => {
            visitor.visit_expr(subexpression);
            visitor.visit_ty(typ)
        }
        ExprKind::Let(pat, expr, _) => {
            visitor.visit_pat(pat);
            visitor.visit_expr(expr);
        }
        ExprKind::If(head_expression, if_block, optional_else) => {
            visitor.visit_expr(head_expression);
            visitor.visit_block(if_block);
            walk_list!(visitor, visit_expr, optional_else);
        }
        ExprKind::While(subexpression, block, opt_label) => {
            walk_list!(visitor, visit_label, opt_label);
            visitor.visit_expr(subexpression);
            visitor.visit_block(block);
        }
        ExprKind::ForLoop(pattern, subexpression, block, opt_label) => {
            walk_list!(visitor, visit_label, opt_label);
            visitor.visit_pat(pattern);
            visitor.visit_expr(subexpression);
            visitor.visit_block(block);
        }
        ExprKind::Loop(block, opt_label, _) => {
            walk_list!(visitor, visit_label, opt_label);
            visitor.visit_block(block);
        }
        ExprKind::Match(subexpression, arms) => {
            visitor.visit_expr(subexpression);
            walk_list!(visitor, visit_arm, arms);
        }
        ExprKind::Closure(box Closure {
            binder,
            capture_clause: _,
            asyncness: _,
            constness: _,
            movability: _,
            fn_decl,
            body,
            fn_decl_span: _,
            fn_arg_span: _,
        }) => {
            visitor.visit_fn(FnKind::Closure(binder, fn_decl, body), expression.span, expression.id)
        }
        ExprKind::Block(block, opt_label) => {
            walk_list!(visitor, visit_label, opt_label);
            visitor.visit_block(block);
        }
        ExprKind::Async(_, body) => {
            visitor.visit_block(body);
        }
        ExprKind::Await(expr) => visitor.visit_expr(expr),
        ExprKind::Assign(lhs, rhs, _) => {
            visitor.visit_expr(lhs);
            visitor.visit_expr(rhs);
        }
        ExprKind::AssignOp(_, left_expression, right_expression) => {
            visitor.visit_expr(left_expression);
            visitor.visit_expr(right_expression);
        }
        ExprKind::Field(subexpression, ident) => {
            visitor.visit_expr(subexpression);
            visitor.visit_ident(*ident);
        }
        ExprKind::Index(main_expression, index_expression) => {
            visitor.visit_expr(main_expression);
            visitor.visit_expr(index_expression)
        }
        ExprKind::Range(start, end, _) => {
            walk_list!(visitor, visit_expr, start);
            walk_list!(visitor, visit_expr, end);
        }
        ExprKind::Underscore => {}
        ExprKind::Path(maybe_qself, path) => {
            if let Some(qself) = maybe_qself {
                visitor.visit_ty(&qself.ty);
            }
            visitor.visit_path(path, expression.id)
        }
        ExprKind::Break(opt_label, opt_expr) => {
            walk_list!(visitor, visit_label, opt_label);
            walk_list!(visitor, visit_expr, opt_expr);
        }
        ExprKind::Continue(opt_label) => {
            walk_list!(visitor, visit_label, opt_label);
        }
        ExprKind::Ret(optional_expression) => {
            walk_list!(visitor, visit_expr, optional_expression);
        }
        ExprKind::Yeet(optional_expression) => {
            walk_list!(visitor, visit_expr, optional_expression);
        }
        ExprKind::MacCall(mac) => visitor.visit_mac_call(mac),
        ExprKind::Paren(subexpression) => visitor.visit_expr(subexpression),
        ExprKind::InlineAsm(asm) => visitor.visit_inline_asm(asm),
        ExprKind::FormatArgs(f) => visitor.visit_format_args(f),
        ExprKind::Yield(optional_expression) => {
            walk_list!(visitor, visit_expr, optional_expression);
        }
        ExprKind::Try(subexpression) => visitor.visit_expr(subexpression),
        ExprKind::TryBlock(body) => visitor.visit_block(body),
        ExprKind::Lit(_) | ExprKind::IncludedBytes(..) | ExprKind::Err => {}
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
    if let VisibilityKind::Restricted { ref path, id, shorthand: _ } = vis.kind {
        visitor.visit_path(path, id);
    }
}

pub fn walk_attribute<'a, V: Visitor<'a>>(visitor: &mut V, attr: &'a Attribute) {
    match &attr.kind {
        AttrKind::Normal(normal) => walk_attr_args(visitor, &normal.item.args),
        AttrKind::DocComment(..) => {}
    }
}

pub fn walk_attr_args<'a, V: Visitor<'a>>(visitor: &mut V, args: &'a AttrArgs) {
    match args {
        AttrArgs::Empty => {}
        AttrArgs::Delimited(_) => {}
        AttrArgs::Eq(_eq_span, AttrArgsEq::Ast(expr)) => visitor.visit_expr(expr),
        AttrArgs::Eq(_, AttrArgsEq::Hir(lit)) => {
            unreachable!("in literal form when walking mac args eq: {:?}", lit)
        }
    }
}
