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

pub use rustc_ast_ir::visit::VisitorResult;
pub use rustc_ast_ir::{try_visit, visit_opt, walk_list, walk_visitable_list};

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
    /// The result type of the `visit_*` methods. Can be either `()`,
    /// or `ControlFlow<T>`.
    type Result: VisitorResult = ();

    fn visit_ident(&mut self, _ident: Ident) -> Self::Result {
        Self::Result::output()
    }
    fn visit_foreign_item(&mut self, i: &'ast ForeignItem) -> Self::Result {
        walk_foreign_item(self, i)
    }
    fn visit_item(&mut self, i: &'ast Item) -> Self::Result {
        walk_item(self, i)
    }
    fn visit_local(&mut self, l: &'ast Local) -> Self::Result {
        walk_local(self, l)
    }
    fn visit_block(&mut self, b: &'ast Block) -> Self::Result {
        walk_block(self, b)
    }
    fn visit_stmt(&mut self, s: &'ast Stmt) -> Self::Result {
        walk_stmt(self, s)
    }
    fn visit_param(&mut self, param: &'ast Param) -> Self::Result {
        walk_param(self, param)
    }
    fn visit_arm(&mut self, a: &'ast Arm) -> Self::Result {
        walk_arm(self, a)
    }
    fn visit_pat(&mut self, p: &'ast Pat) -> Self::Result {
        walk_pat(self, p)
    }
    fn visit_anon_const(&mut self, c: &'ast AnonConst) -> Self::Result {
        walk_anon_const(self, c)
    }
    fn visit_expr(&mut self, ex: &'ast Expr) -> Self::Result {
        walk_expr(self, ex)
    }
    /// This method is a hack to workaround unstable of `stmt_expr_attributes`.
    /// It can be removed once that feature is stabilized.
    fn visit_method_receiver_expr(&mut self, ex: &'ast Expr) -> Self::Result {
        self.visit_expr(ex)
    }
    fn visit_expr_post(&mut self, _ex: &'ast Expr) -> Self::Result {
        Self::Result::output()
    }
    fn visit_ty(&mut self, t: &'ast Ty) -> Self::Result {
        walk_ty(self, t)
    }
    fn visit_generic_param(&mut self, param: &'ast GenericParam) -> Self::Result {
        walk_generic_param(self, param)
    }
    fn visit_generics(&mut self, g: &'ast Generics) -> Self::Result {
        walk_generics(self, g)
    }
    fn visit_closure_binder(&mut self, b: &'ast ClosureBinder) -> Self::Result {
        walk_closure_binder(self, b)
    }
    fn visit_where_predicate(&mut self, p: &'ast WherePredicate) -> Self::Result {
        walk_where_predicate(self, p)
    }
    fn visit_fn(&mut self, fk: FnKind<'ast>, _: Span, _: NodeId) -> Self::Result {
        walk_fn(self, fk)
    }
    fn visit_assoc_item(&mut self, i: &'ast AssocItem, ctxt: AssocCtxt) -> Self::Result {
        walk_assoc_item(self, i, ctxt)
    }
    fn visit_trait_ref(&mut self, t: &'ast TraitRef) -> Self::Result {
        walk_trait_ref(self, t)
    }
    fn visit_param_bound(&mut self, bounds: &'ast GenericBound, _ctxt: BoundKind) -> Self::Result {
        walk_param_bound(self, bounds)
    }
    fn visit_poly_trait_ref(&mut self, t: &'ast PolyTraitRef) -> Self::Result {
        walk_poly_trait_ref(self, t)
    }
    fn visit_variant_data(&mut self, s: &'ast VariantData) -> Self::Result {
        walk_struct_def(self, s)
    }
    fn visit_field_def(&mut self, s: &'ast FieldDef) -> Self::Result {
        walk_field_def(self, s)
    }
    fn visit_enum_def(&mut self, enum_definition: &'ast EnumDef) -> Self::Result {
        walk_enum_def(self, enum_definition)
    }
    fn visit_variant(&mut self, v: &'ast Variant) -> Self::Result {
        walk_variant(self, v)
    }
    fn visit_variant_discr(&mut self, discr: &'ast AnonConst) -> Self::Result {
        self.visit_anon_const(discr)
    }
    fn visit_label(&mut self, label: &'ast Label) -> Self::Result {
        walk_label(self, label)
    }
    fn visit_lifetime(&mut self, lifetime: &'ast Lifetime, _: LifetimeCtxt) -> Self::Result {
        walk_lifetime(self, lifetime)
    }
    fn visit_mac_call(&mut self, mac: &'ast MacCall) -> Self::Result {
        walk_mac(self, mac)
    }
    fn visit_mac_def(&mut self, _mac: &'ast MacroDef, _id: NodeId) -> Self::Result {
        Self::Result::output()
    }
    fn visit_path(&mut self, path: &'ast Path, _id: NodeId) -> Self::Result {
        walk_path(self, path)
    }
    fn visit_use_tree(
        &mut self,
        use_tree: &'ast UseTree,
        id: NodeId,
        _nested: bool,
    ) -> Self::Result {
        walk_use_tree(self, use_tree, id)
    }
    fn visit_path_segment(&mut self, path_segment: &'ast PathSegment) -> Self::Result {
        walk_path_segment(self, path_segment)
    }
    fn visit_generic_args(&mut self, generic_args: &'ast GenericArgs) -> Self::Result {
        walk_generic_args(self, generic_args)
    }
    fn visit_generic_arg(&mut self, generic_arg: &'ast GenericArg) -> Self::Result {
        walk_generic_arg(self, generic_arg)
    }
    fn visit_assoc_constraint(&mut self, constraint: &'ast AssocConstraint) -> Self::Result {
        walk_assoc_constraint(self, constraint)
    }
    fn visit_attribute(&mut self, attr: &'ast Attribute) -> Self::Result {
        walk_attribute(self, attr)
    }
    fn visit_vis(&mut self, vis: &'ast Visibility) -> Self::Result {
        walk_vis(self, vis)
    }
    fn visit_fn_ret_ty(&mut self, ret_ty: &'ast FnRetTy) -> Self::Result {
        walk_fn_ret_ty(self, ret_ty)
    }
    fn visit_fn_header(&mut self, _header: &'ast FnHeader) -> Self::Result {
        Self::Result::output()
    }
    fn visit_expr_field(&mut self, f: &'ast ExprField) -> Self::Result {
        walk_expr_field(self, f)
    }
    fn visit_pat_field(&mut self, fp: &'ast PatField) -> Self::Result {
        walk_pat_field(self, fp)
    }
    fn visit_crate(&mut self, krate: &'ast Crate) -> Self::Result {
        walk_crate(self, krate)
    }
    fn visit_inline_asm(&mut self, asm: &'ast InlineAsm) -> Self::Result {
        walk_inline_asm(self, asm)
    }
    fn visit_format_args(&mut self, fmt: &'ast FormatArgs) -> Self::Result {
        walk_format_args(self, fmt)
    }
    fn visit_inline_asm_sym(&mut self, sym: &'ast InlineAsmSym) -> Self::Result {
        walk_inline_asm_sym(self, sym)
    }
    fn visit_capture_by(&mut self, _capture_by: &'ast CaptureBy) -> Self::Result {
        Self::Result::output()
    }
}

pub fn walk_crate<'a, V: Visitor<'a>>(visitor: &mut V, krate: &'a Crate) -> V::Result {
    walk_list!(visitor, visit_item, &krate.items);
    walk_list!(visitor, visit_attribute, &krate.attrs);
    V::Result::output()
}

pub fn walk_local<'a, V: Visitor<'a>>(visitor: &mut V, local: &'a Local) -> V::Result {
    walk_list!(visitor, visit_attribute, &local.attrs);
    try_visit!(visitor.visit_pat(&local.pat));
    visit_opt!(visitor, visit_ty, &local.ty);
    if let Some((init, els)) = local.kind.init_else_opt() {
        try_visit!(visitor.visit_expr(init));
        visit_opt!(visitor, visit_block, els);
    }
    V::Result::output()
}

pub fn walk_label<'a, V: Visitor<'a>>(visitor: &mut V, label: &'a Label) -> V::Result {
    visitor.visit_ident(label.ident)
}

pub fn walk_lifetime<'a, V: Visitor<'a>>(visitor: &mut V, lifetime: &'a Lifetime) -> V::Result {
    visitor.visit_ident(lifetime.ident)
}

pub fn walk_poly_trait_ref<'a, V>(visitor: &mut V, trait_ref: &'a PolyTraitRef) -> V::Result
where
    V: Visitor<'a>,
{
    walk_list!(visitor, visit_generic_param, &trait_ref.bound_generic_params);
    visitor.visit_trait_ref(&trait_ref.trait_ref)
}

pub fn walk_trait_ref<'a, V: Visitor<'a>>(visitor: &mut V, trait_ref: &'a TraitRef) -> V::Result {
    visitor.visit_path(&trait_ref.path, trait_ref.ref_id)
}

pub fn walk_item<'a, V: Visitor<'a>>(visitor: &mut V, item: &'a Item) -> V::Result {
    try_visit!(visitor.visit_vis(&item.vis));
    try_visit!(visitor.visit_ident(item.ident));
    match &item.kind {
        ItemKind::ExternCrate(_) => {}
        ItemKind::Use(use_tree) => try_visit!(visitor.visit_use_tree(use_tree, item.id, false)),
        ItemKind::Static(box StaticItem { ty, mutability: _, expr }) => {
            try_visit!(visitor.visit_ty(ty));
            visit_opt!(visitor, visit_expr, expr);
        }
        ItemKind::Const(box ConstItem { defaultness: _, generics, ty, expr }) => {
            try_visit!(visitor.visit_generics(generics));
            try_visit!(visitor.visit_ty(ty));
            visit_opt!(visitor, visit_expr, expr);
        }
        ItemKind::Fn(box Fn { defaultness: _, generics, sig, body }) => {
            let kind =
                FnKind::Fn(FnCtxt::Free, item.ident, sig, &item.vis, generics, body.as_deref());
            try_visit!(visitor.visit_fn(kind, item.span, item.id));
        }
        ItemKind::Mod(_unsafety, mod_kind) => match mod_kind {
            ModKind::Loaded(items, _inline, _inner_span) => {
                walk_list!(visitor, visit_item, items);
            }
            ModKind::Unloaded => {}
        },
        ItemKind::ForeignMod(foreign_module) => {
            walk_list!(visitor, visit_foreign_item, &foreign_module.items);
        }
        ItemKind::GlobalAsm(asm) => try_visit!(visitor.visit_inline_asm(asm)),
        ItemKind::TyAlias(box TyAlias { generics, bounds, ty, .. }) => {
            try_visit!(visitor.visit_generics(generics));
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
            visit_opt!(visitor, visit_ty, ty);
        }
        ItemKind::Enum(enum_definition, generics) => {
            try_visit!(visitor.visit_generics(generics));
            try_visit!(visitor.visit_enum_def(enum_definition));
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
            try_visit!(visitor.visit_generics(generics));
            visit_opt!(visitor, visit_trait_ref, of_trait);
            try_visit!(visitor.visit_ty(self_ty));
            walk_list!(visitor, visit_assoc_item, items, AssocCtxt::Impl);
        }
        ItemKind::Struct(struct_definition, generics)
        | ItemKind::Union(struct_definition, generics) => {
            try_visit!(visitor.visit_generics(generics));
            try_visit!(visitor.visit_variant_data(struct_definition));
        }
        ItemKind::Trait(box Trait { unsafety: _, is_auto: _, generics, bounds, items }) => {
            try_visit!(visitor.visit_generics(generics));
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::SuperTraits);
            walk_list!(visitor, visit_assoc_item, items, AssocCtxt::Trait);
        }
        ItemKind::TraitAlias(generics, bounds) => {
            try_visit!(visitor.visit_generics(generics));
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
        }
        ItemKind::MacCall(mac) => try_visit!(visitor.visit_mac_call(mac)),
        ItemKind::MacroDef(ts) => try_visit!(visitor.visit_mac_def(ts, item.id)),
        ItemKind::Delegation(box Delegation { id, qself, path, body }) => {
            if let Some(qself) = qself {
                try_visit!(visitor.visit_ty(&qself.ty));
            }
            try_visit!(visitor.visit_path(path, *id));
            visit_opt!(visitor, visit_block, body);
        }
    }
    walk_list!(visitor, visit_attribute, &item.attrs);
    V::Result::output()
}

pub fn walk_enum_def<'a, V: Visitor<'a>>(
    visitor: &mut V,
    enum_definition: &'a EnumDef,
) -> V::Result {
    walk_list!(visitor, visit_variant, &enum_definition.variants);
    V::Result::output()
}

pub fn walk_variant<'a, V: Visitor<'a>>(visitor: &mut V, variant: &'a Variant) -> V::Result
where
    V: Visitor<'a>,
{
    try_visit!(visitor.visit_ident(variant.ident));
    try_visit!(visitor.visit_vis(&variant.vis));
    try_visit!(visitor.visit_variant_data(&variant.data));
    visit_opt!(visitor, visit_variant_discr, &variant.disr_expr);
    walk_list!(visitor, visit_attribute, &variant.attrs);
    V::Result::output()
}

pub fn walk_expr_field<'a, V: Visitor<'a>>(visitor: &mut V, f: &'a ExprField) -> V::Result {
    try_visit!(visitor.visit_expr(&f.expr));
    try_visit!(visitor.visit_ident(f.ident));
    walk_list!(visitor, visit_attribute, &f.attrs);
    V::Result::output()
}

pub fn walk_pat_field<'a, V: Visitor<'a>>(visitor: &mut V, fp: &'a PatField) -> V::Result {
    try_visit!(visitor.visit_ident(fp.ident));
    try_visit!(visitor.visit_pat(&fp.pat));
    walk_list!(visitor, visit_attribute, &fp.attrs);
    V::Result::output()
}

pub fn walk_ty<'a, V: Visitor<'a>>(visitor: &mut V, typ: &'a Ty) -> V::Result {
    match &typ.kind {
        TyKind::Slice(ty) | TyKind::Paren(ty) => try_visit!(visitor.visit_ty(ty)),
        TyKind::Ptr(mutable_type) => try_visit!(visitor.visit_ty(&mutable_type.ty)),
        TyKind::Ref(opt_lifetime, mutable_type) => {
            visit_opt!(visitor, visit_lifetime, opt_lifetime, LifetimeCtxt::Ref);
            try_visit!(visitor.visit_ty(&mutable_type.ty));
        }
        TyKind::Tup(tuple_element_types) => {
            walk_list!(visitor, visit_ty, tuple_element_types);
        }
        TyKind::BareFn(function_declaration) => {
            walk_list!(visitor, visit_generic_param, &function_declaration.generic_params);
            try_visit!(walk_fn_decl(visitor, &function_declaration.decl));
        }
        TyKind::Path(maybe_qself, path) => {
            if let Some(qself) = maybe_qself {
                try_visit!(visitor.visit_ty(&qself.ty));
            }
            try_visit!(visitor.visit_path(path, typ.id));
        }
        TyKind::Array(ty, length) => {
            try_visit!(visitor.visit_ty(ty));
            try_visit!(visitor.visit_anon_const(length));
        }
        TyKind::TraitObject(bounds, ..) => {
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::TraitObject);
        }
        TyKind::ImplTrait(_, bounds) => {
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::Impl);
        }
        TyKind::Typeof(expression) => try_visit!(visitor.visit_anon_const(expression)),
        TyKind::Infer | TyKind::ImplicitSelf | TyKind::Dummy | TyKind::Err(_) => {}
        TyKind::MacCall(mac) => try_visit!(visitor.visit_mac_call(mac)),
        TyKind::Never | TyKind::CVarArgs => {}
        TyKind::AnonStruct(_, ref fields) | TyKind::AnonUnion(_, ref fields) => {
            walk_list!(visitor, visit_field_def, fields);
        }
    }
    V::Result::output()
}

pub fn walk_path<'a, V: Visitor<'a>>(visitor: &mut V, path: &'a Path) -> V::Result {
    walk_list!(visitor, visit_path_segment, &path.segments);
    V::Result::output()
}

pub fn walk_use_tree<'a, V: Visitor<'a>>(
    visitor: &mut V,
    use_tree: &'a UseTree,
    id: NodeId,
) -> V::Result {
    try_visit!(visitor.visit_path(&use_tree.prefix, id));
    match use_tree.kind {
        UseTreeKind::Simple(rename) => {
            // The extra IDs are handled during HIR lowering.
            visit_opt!(visitor, visit_ident, rename);
        }
        UseTreeKind::Glob => {}
        UseTreeKind::Nested(ref use_trees) => {
            for &(ref nested_tree, nested_id) in use_trees {
                try_visit!(visitor.visit_use_tree(nested_tree, nested_id, true));
            }
        }
    }
    V::Result::output()
}

pub fn walk_path_segment<'a, V: Visitor<'a>>(
    visitor: &mut V,
    segment: &'a PathSegment,
) -> V::Result {
    try_visit!(visitor.visit_ident(segment.ident));
    visit_opt!(visitor, visit_generic_args, &segment.args);
    V::Result::output()
}

pub fn walk_generic_args<'a, V>(visitor: &mut V, generic_args: &'a GenericArgs) -> V::Result
where
    V: Visitor<'a>,
{
    match generic_args {
        GenericArgs::AngleBracketed(data) => {
            for arg in &data.args {
                match arg {
                    AngleBracketedArg::Arg(a) => try_visit!(visitor.visit_generic_arg(a)),
                    AngleBracketedArg::Constraint(c) => {
                        try_visit!(visitor.visit_assoc_constraint(c))
                    }
                }
            }
        }
        GenericArgs::Parenthesized(data) => {
            walk_list!(visitor, visit_ty, &data.inputs);
            try_visit!(visitor.visit_fn_ret_ty(&data.output));
        }
    }
    V::Result::output()
}

pub fn walk_generic_arg<'a, V>(visitor: &mut V, generic_arg: &'a GenericArg) -> V::Result
where
    V: Visitor<'a>,
{
    match generic_arg {
        GenericArg::Lifetime(lt) => visitor.visit_lifetime(lt, LifetimeCtxt::GenericArg),
        GenericArg::Type(ty) => visitor.visit_ty(ty),
        GenericArg::Const(ct) => visitor.visit_anon_const(ct),
    }
}

pub fn walk_assoc_constraint<'a, V: Visitor<'a>>(
    visitor: &mut V,
    constraint: &'a AssocConstraint,
) -> V::Result {
    try_visit!(visitor.visit_ident(constraint.ident));
    visit_opt!(visitor, visit_generic_args, &constraint.gen_args);
    match &constraint.kind {
        AssocConstraintKind::Equality { term } => match term {
            Term::Ty(ty) => try_visit!(visitor.visit_ty(ty)),
            Term::Const(c) => try_visit!(visitor.visit_anon_const(c)),
        },
        AssocConstraintKind::Bound { bounds } => {
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
        }
    }
    V::Result::output()
}

pub fn walk_pat<'a, V: Visitor<'a>>(visitor: &mut V, pattern: &'a Pat) -> V::Result {
    match &pattern.kind {
        PatKind::TupleStruct(opt_qself, path, elems) => {
            if let Some(qself) = opt_qself {
                try_visit!(visitor.visit_ty(&qself.ty));
            }
            try_visit!(visitor.visit_path(path, pattern.id));
            walk_list!(visitor, visit_pat, elems);
        }
        PatKind::Path(opt_qself, path) => {
            if let Some(qself) = opt_qself {
                try_visit!(visitor.visit_ty(&qself.ty));
            }
            try_visit!(visitor.visit_path(path, pattern.id))
        }
        PatKind::Struct(opt_qself, path, fields, _) => {
            if let Some(qself) = opt_qself {
                try_visit!(visitor.visit_ty(&qself.ty));
            }
            try_visit!(visitor.visit_path(path, pattern.id));
            walk_list!(visitor, visit_pat_field, fields);
        }
        PatKind::Box(subpattern) | PatKind::Ref(subpattern, _) | PatKind::Paren(subpattern) => {
            try_visit!(visitor.visit_pat(subpattern));
        }
        PatKind::Ident(_, ident, optional_subpattern) => {
            try_visit!(visitor.visit_ident(*ident));
            visit_opt!(visitor, visit_pat, optional_subpattern);
        }
        PatKind::Lit(expression) => try_visit!(visitor.visit_expr(expression)),
        PatKind::Range(lower_bound, upper_bound, _) => {
            visit_opt!(visitor, visit_expr, lower_bound);
            visit_opt!(visitor, visit_expr, upper_bound);
        }
        PatKind::Wild | PatKind::Rest | PatKind::Never | PatKind::Err(_) => {}
        PatKind::Tuple(elems) | PatKind::Slice(elems) | PatKind::Or(elems) => {
            walk_list!(visitor, visit_pat, elems);
        }
        PatKind::MacCall(mac) => try_visit!(visitor.visit_mac_call(mac)),
    }
    V::Result::output()
}

pub fn walk_foreign_item<'a, V: Visitor<'a>>(visitor: &mut V, item: &'a ForeignItem) -> V::Result {
    let &Item { id, span, ident, ref vis, ref attrs, ref kind, tokens: _ } = item;
    try_visit!(visitor.visit_vis(vis));
    try_visit!(visitor.visit_ident(ident));
    walk_list!(visitor, visit_attribute, attrs);
    match kind {
        ForeignItemKind::Static(ty, _, expr) => {
            try_visit!(visitor.visit_ty(ty));
            visit_opt!(visitor, visit_expr, expr);
        }
        ForeignItemKind::Fn(box Fn { defaultness: _, generics, sig, body }) => {
            let kind = FnKind::Fn(FnCtxt::Foreign, ident, sig, vis, generics, body.as_deref());
            try_visit!(visitor.visit_fn(kind, span, id));
        }
        ForeignItemKind::TyAlias(box TyAlias { generics, bounds, ty, .. }) => {
            try_visit!(visitor.visit_generics(generics));
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
            visit_opt!(visitor, visit_ty, ty);
        }
        ForeignItemKind::MacCall(mac) => {
            try_visit!(visitor.visit_mac_call(mac));
        }
    }
    V::Result::output()
}

pub fn walk_param_bound<'a, V: Visitor<'a>>(visitor: &mut V, bound: &'a GenericBound) -> V::Result {
    match bound {
        GenericBound::Trait(typ, _modifier) => visitor.visit_poly_trait_ref(typ),
        GenericBound::Outlives(lifetime) => visitor.visit_lifetime(lifetime, LifetimeCtxt::Bound),
    }
}

pub fn walk_generic_param<'a, V: Visitor<'a>>(
    visitor: &mut V,
    param: &'a GenericParam,
) -> V::Result {
    try_visit!(visitor.visit_ident(param.ident));
    walk_list!(visitor, visit_attribute, &param.attrs);
    walk_list!(visitor, visit_param_bound, &param.bounds, BoundKind::Bound);
    match &param.kind {
        GenericParamKind::Lifetime => (),
        GenericParamKind::Type { default } => visit_opt!(visitor, visit_ty, default),
        GenericParamKind::Const { ty, default, .. } => {
            try_visit!(visitor.visit_ty(ty));
            visit_opt!(visitor, visit_anon_const, default);
        }
    }
    V::Result::output()
}

pub fn walk_generics<'a, V: Visitor<'a>>(visitor: &mut V, generics: &'a Generics) -> V::Result {
    walk_list!(visitor, visit_generic_param, &generics.params);
    walk_list!(visitor, visit_where_predicate, &generics.where_clause.predicates);
    V::Result::output()
}

pub fn walk_closure_binder<'a, V: Visitor<'a>>(
    visitor: &mut V,
    binder: &'a ClosureBinder,
) -> V::Result {
    match binder {
        ClosureBinder::NotPresent => {}
        ClosureBinder::For { generic_params, span: _ } => {
            walk_list!(visitor, visit_generic_param, generic_params)
        }
    }
    V::Result::output()
}

pub fn walk_where_predicate<'a, V: Visitor<'a>>(
    visitor: &mut V,
    predicate: &'a WherePredicate,
) -> V::Result {
    match predicate {
        WherePredicate::BoundPredicate(WhereBoundPredicate {
            bounded_ty,
            bounds,
            bound_generic_params,
            ..
        }) => {
            try_visit!(visitor.visit_ty(bounded_ty));
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
            walk_list!(visitor, visit_generic_param, bound_generic_params);
        }
        WherePredicate::RegionPredicate(WhereRegionPredicate { lifetime, bounds, .. }) => {
            try_visit!(visitor.visit_lifetime(lifetime, LifetimeCtxt::Bound));
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
        }
        WherePredicate::EqPredicate(WhereEqPredicate { lhs_ty, rhs_ty, .. }) => {
            try_visit!(visitor.visit_ty(lhs_ty));
            try_visit!(visitor.visit_ty(rhs_ty));
        }
    }
    V::Result::output()
}

pub fn walk_fn_ret_ty<'a, V: Visitor<'a>>(visitor: &mut V, ret_ty: &'a FnRetTy) -> V::Result {
    if let FnRetTy::Ty(output_ty) = ret_ty {
        try_visit!(visitor.visit_ty(output_ty));
    }
    V::Result::output()
}

pub fn walk_fn_decl<'a, V: Visitor<'a>>(
    visitor: &mut V,
    function_declaration: &'a FnDecl,
) -> V::Result {
    walk_list!(visitor, visit_param, &function_declaration.inputs);
    visitor.visit_fn_ret_ty(&function_declaration.output)
}

pub fn walk_fn<'a, V: Visitor<'a>>(visitor: &mut V, kind: FnKind<'a>) -> V::Result {
    match kind {
        FnKind::Fn(_, _, sig, _, generics, body) => {
            try_visit!(visitor.visit_generics(generics));
            try_visit!(visitor.visit_fn_header(&sig.header));
            try_visit!(walk_fn_decl(visitor, &sig.decl));
            visit_opt!(visitor, visit_block, body);
        }
        FnKind::Closure(binder, decl, body) => {
            try_visit!(visitor.visit_closure_binder(binder));
            try_visit!(walk_fn_decl(visitor, decl));
            try_visit!(visitor.visit_expr(body));
        }
    }
    V::Result::output()
}

pub fn walk_assoc_item<'a, V: Visitor<'a>>(
    visitor: &mut V,
    item: &'a AssocItem,
    ctxt: AssocCtxt,
) -> V::Result {
    let &Item { id, span, ident, ref vis, ref attrs, ref kind, tokens: _ } = item;
    try_visit!(visitor.visit_vis(vis));
    try_visit!(visitor.visit_ident(ident));
    walk_list!(visitor, visit_attribute, attrs);
    match kind {
        AssocItemKind::Const(box ConstItem { defaultness: _, generics, ty, expr }) => {
            try_visit!(visitor.visit_generics(generics));
            try_visit!(visitor.visit_ty(ty));
            visit_opt!(visitor, visit_expr, expr);
        }
        AssocItemKind::Fn(box Fn { defaultness: _, generics, sig, body }) => {
            let kind = FnKind::Fn(FnCtxt::Assoc(ctxt), ident, sig, vis, generics, body.as_deref());
            try_visit!(visitor.visit_fn(kind, span, id));
        }
        AssocItemKind::Type(box TyAlias { generics, bounds, ty, .. }) => {
            try_visit!(visitor.visit_generics(generics));
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
            visit_opt!(visitor, visit_ty, ty);
        }
        AssocItemKind::MacCall(mac) => {
            try_visit!(visitor.visit_mac_call(mac));
        }
        AssocItemKind::Delegation(box Delegation { id, qself, path, body }) => {
            if let Some(qself) = qself {
                visitor.visit_ty(&qself.ty);
            }
            try_visit!(visitor.visit_path(path, *id));
            visit_opt!(visitor, visit_block, body);
        }
    }
    V::Result::output()
}

pub fn walk_struct_def<'a, V: Visitor<'a>>(
    visitor: &mut V,
    struct_definition: &'a VariantData,
) -> V::Result {
    walk_list!(visitor, visit_field_def, struct_definition.fields());
    V::Result::output()
}

pub fn walk_field_def<'a, V: Visitor<'a>>(visitor: &mut V, field: &'a FieldDef) -> V::Result {
    try_visit!(visitor.visit_vis(&field.vis));
    visit_opt!(visitor, visit_ident, field.ident);
    try_visit!(visitor.visit_ty(&field.ty));
    walk_list!(visitor, visit_attribute, &field.attrs);
    V::Result::output()
}

pub fn walk_block<'a, V: Visitor<'a>>(visitor: &mut V, block: &'a Block) -> V::Result {
    walk_list!(visitor, visit_stmt, &block.stmts);
    V::Result::output()
}

pub fn walk_stmt<'a, V: Visitor<'a>>(visitor: &mut V, statement: &'a Stmt) -> V::Result {
    match &statement.kind {
        StmtKind::Local(local) => try_visit!(visitor.visit_local(local)),
        StmtKind::Item(item) => try_visit!(visitor.visit_item(item)),
        StmtKind::Expr(expr) | StmtKind::Semi(expr) => try_visit!(visitor.visit_expr(expr)),
        StmtKind::Empty => {}
        StmtKind::MacCall(mac) => {
            let MacCallStmt { mac, attrs, style: _, tokens: _ } = &**mac;
            try_visit!(visitor.visit_mac_call(mac));
            walk_list!(visitor, visit_attribute, attrs);
        }
    }
    V::Result::output()
}

pub fn walk_mac<'a, V: Visitor<'a>>(visitor: &mut V, mac: &'a MacCall) -> V::Result {
    visitor.visit_path(&mac.path, DUMMY_NODE_ID)
}

pub fn walk_anon_const<'a, V: Visitor<'a>>(visitor: &mut V, constant: &'a AnonConst) -> V::Result {
    visitor.visit_expr(&constant.value)
}

pub fn walk_inline_asm<'a, V: Visitor<'a>>(visitor: &mut V, asm: &'a InlineAsm) -> V::Result {
    for (op, _) in &asm.operands {
        match op {
            InlineAsmOperand::In { expr, .. }
            | InlineAsmOperand::Out { expr: Some(expr), .. }
            | InlineAsmOperand::InOut { expr, .. } => try_visit!(visitor.visit_expr(expr)),
            InlineAsmOperand::Out { expr: None, .. } => {}
            InlineAsmOperand::SplitInOut { in_expr, out_expr, .. } => {
                try_visit!(visitor.visit_expr(in_expr));
                visit_opt!(visitor, visit_expr, out_expr);
            }
            InlineAsmOperand::Const { anon_const, .. } => {
                try_visit!(visitor.visit_anon_const(anon_const))
            }
            InlineAsmOperand::Sym { sym } => try_visit!(visitor.visit_inline_asm_sym(sym)),
        }
    }
    V::Result::output()
}

pub fn walk_inline_asm_sym<'a, V: Visitor<'a>>(
    visitor: &mut V,
    sym: &'a InlineAsmSym,
) -> V::Result {
    if let Some(qself) = &sym.qself {
        try_visit!(visitor.visit_ty(&qself.ty));
    }
    visitor.visit_path(&sym.path, sym.id)
}

pub fn walk_format_args<'a, V: Visitor<'a>>(visitor: &mut V, fmt: &'a FormatArgs) -> V::Result {
    for arg in fmt.arguments.all_args() {
        if let FormatArgumentKind::Named(name) = arg.kind {
            try_visit!(visitor.visit_ident(name));
        }
        try_visit!(visitor.visit_expr(&arg.expr));
    }
    V::Result::output()
}

pub fn walk_expr<'a, V: Visitor<'a>>(visitor: &mut V, expression: &'a Expr) -> V::Result {
    walk_list!(visitor, visit_attribute, &expression.attrs);

    match &expression.kind {
        ExprKind::Array(subexpressions) => {
            walk_list!(visitor, visit_expr, subexpressions);
        }
        ExprKind::ConstBlock(anon_const) => try_visit!(visitor.visit_anon_const(anon_const)),
        ExprKind::Repeat(element, count) => {
            try_visit!(visitor.visit_expr(element));
            try_visit!(visitor.visit_anon_const(count));
        }
        ExprKind::Struct(se) => {
            if let Some(qself) = &se.qself {
                try_visit!(visitor.visit_ty(&qself.ty));
            }
            try_visit!(visitor.visit_path(&se.path, expression.id));
            walk_list!(visitor, visit_expr_field, &se.fields);
            match &se.rest {
                StructRest::Base(expr) => try_visit!(visitor.visit_expr(expr)),
                StructRest::Rest(_span) => {}
                StructRest::None => {}
            }
        }
        ExprKind::Tup(subexpressions) => {
            walk_list!(visitor, visit_expr, subexpressions);
        }
        ExprKind::Call(callee_expression, arguments) => {
            try_visit!(visitor.visit_expr(callee_expression));
            walk_list!(visitor, visit_expr, arguments);
        }
        ExprKind::MethodCall(box MethodCall { seg, receiver, args, span: _ }) => {
            try_visit!(visitor.visit_path_segment(seg));
            try_visit!(visitor.visit_expr(receiver));
            walk_list!(visitor, visit_expr, args);
        }
        ExprKind::Binary(_, left_expression, right_expression) => {
            try_visit!(visitor.visit_expr(left_expression));
            try_visit!(visitor.visit_expr(right_expression));
        }
        ExprKind::AddrOf(_, _, subexpression) | ExprKind::Unary(_, subexpression) => {
            try_visit!(visitor.visit_expr(subexpression));
        }
        ExprKind::Cast(subexpression, typ) | ExprKind::Type(subexpression, typ) => {
            try_visit!(visitor.visit_expr(subexpression));
            try_visit!(visitor.visit_ty(typ));
        }
        ExprKind::Let(pat, expr, _, _) => {
            try_visit!(visitor.visit_pat(pat));
            try_visit!(visitor.visit_expr(expr));
        }
        ExprKind::If(head_expression, if_block, optional_else) => {
            try_visit!(visitor.visit_expr(head_expression));
            try_visit!(visitor.visit_block(if_block));
            visit_opt!(visitor, visit_expr, optional_else);
        }
        ExprKind::While(subexpression, block, opt_label) => {
            visit_opt!(visitor, visit_label, opt_label);
            try_visit!(visitor.visit_expr(subexpression));
            try_visit!(visitor.visit_block(block));
        }
        ExprKind::ForLoop { pat, iter, body, label, kind: _ } => {
            visit_opt!(visitor, visit_label, label);
            try_visit!(visitor.visit_pat(pat));
            try_visit!(visitor.visit_expr(iter));
            try_visit!(visitor.visit_block(body));
        }
        ExprKind::Loop(block, opt_label, _) => {
            visit_opt!(visitor, visit_label, opt_label);
            try_visit!(visitor.visit_block(block));
        }
        ExprKind::Match(subexpression, arms) => {
            try_visit!(visitor.visit_expr(subexpression));
            walk_list!(visitor, visit_arm, arms);
        }
        ExprKind::Closure(box Closure {
            binder,
            capture_clause,
            coroutine_kind: _,
            constness: _,
            movability: _,
            fn_decl,
            body,
            fn_decl_span: _,
            fn_arg_span: _,
        }) => {
            try_visit!(visitor.visit_capture_by(capture_clause));
            try_visit!(visitor.visit_fn(
                FnKind::Closure(binder, fn_decl, body),
                expression.span,
                expression.id
            ))
        }
        ExprKind::Block(block, opt_label) => {
            visit_opt!(visitor, visit_label, opt_label);
            try_visit!(visitor.visit_block(block));
        }
        ExprKind::Gen(_, body, _) => try_visit!(visitor.visit_block(body)),
        ExprKind::Await(expr, _) => try_visit!(visitor.visit_expr(expr)),
        ExprKind::Assign(lhs, rhs, _) => {
            try_visit!(visitor.visit_expr(lhs));
            try_visit!(visitor.visit_expr(rhs));
        }
        ExprKind::AssignOp(_, left_expression, right_expression) => {
            try_visit!(visitor.visit_expr(left_expression));
            try_visit!(visitor.visit_expr(right_expression));
        }
        ExprKind::Field(subexpression, ident) => {
            try_visit!(visitor.visit_expr(subexpression));
            try_visit!(visitor.visit_ident(*ident));
        }
        ExprKind::Index(main_expression, index_expression, _) => {
            try_visit!(visitor.visit_expr(main_expression));
            try_visit!(visitor.visit_expr(index_expression));
        }
        ExprKind::Range(start, end, _) => {
            visit_opt!(visitor, visit_expr, start);
            visit_opt!(visitor, visit_expr, end);
        }
        ExprKind::Underscore => {}
        ExprKind::Path(maybe_qself, path) => {
            if let Some(qself) = maybe_qself {
                try_visit!(visitor.visit_ty(&qself.ty));
            }
            try_visit!(visitor.visit_path(path, expression.id));
        }
        ExprKind::Break(opt_label, opt_expr) => {
            visit_opt!(visitor, visit_label, opt_label);
            visit_opt!(visitor, visit_expr, opt_expr);
        }
        ExprKind::Continue(opt_label) => {
            visit_opt!(visitor, visit_label, opt_label);
        }
        ExprKind::Ret(optional_expression) => {
            visit_opt!(visitor, visit_expr, optional_expression);
        }
        ExprKind::Yeet(optional_expression) => {
            visit_opt!(visitor, visit_expr, optional_expression);
        }
        ExprKind::Become(expr) => try_visit!(visitor.visit_expr(expr)),
        ExprKind::MacCall(mac) => try_visit!(visitor.visit_mac_call(mac)),
        ExprKind::Paren(subexpression) => try_visit!(visitor.visit_expr(subexpression)),
        ExprKind::InlineAsm(asm) => try_visit!(visitor.visit_inline_asm(asm)),
        ExprKind::FormatArgs(f) => try_visit!(visitor.visit_format_args(f)),
        ExprKind::OffsetOf(container, fields) => {
            visitor.visit_ty(container);
            walk_list!(visitor, visit_ident, fields.iter().copied());
        }
        ExprKind::Yield(optional_expression) => {
            visit_opt!(visitor, visit_expr, optional_expression);
        }
        ExprKind::Try(subexpression) => try_visit!(visitor.visit_expr(subexpression)),
        ExprKind::TryBlock(body) => try_visit!(visitor.visit_block(body)),
        ExprKind::Lit(_) | ExprKind::IncludedBytes(..) | ExprKind::Err(_) | ExprKind::Dummy => {}
    }

    visitor.visit_expr_post(expression)
}

pub fn walk_param<'a, V: Visitor<'a>>(visitor: &mut V, param: &'a Param) -> V::Result {
    walk_list!(visitor, visit_attribute, &param.attrs);
    try_visit!(visitor.visit_pat(&param.pat));
    try_visit!(visitor.visit_ty(&param.ty));
    V::Result::output()
}

pub fn walk_arm<'a, V: Visitor<'a>>(visitor: &mut V, arm: &'a Arm) -> V::Result {
    try_visit!(visitor.visit_pat(&arm.pat));
    visit_opt!(visitor, visit_expr, &arm.guard);
    visit_opt!(visitor, visit_expr, &arm.body);
    walk_list!(visitor, visit_attribute, &arm.attrs);
    V::Result::output()
}

pub fn walk_vis<'a, V: Visitor<'a>>(visitor: &mut V, vis: &'a Visibility) -> V::Result {
    if let VisibilityKind::Restricted { ref path, id, shorthand: _ } = vis.kind {
        try_visit!(visitor.visit_path(path, id));
    }
    V::Result::output()
}

pub fn walk_attribute<'a, V: Visitor<'a>>(visitor: &mut V, attr: &'a Attribute) -> V::Result {
    match &attr.kind {
        AttrKind::Normal(normal) => try_visit!(walk_attr_args(visitor, &normal.item.args)),
        AttrKind::DocComment(..) => {}
    }
    V::Result::output()
}

pub fn walk_attr_args<'a, V: Visitor<'a>>(visitor: &mut V, args: &'a AttrArgs) -> V::Result {
    match args {
        AttrArgs::Empty => {}
        AttrArgs::Delimited(_) => {}
        AttrArgs::Eq(_eq_span, AttrArgsEq::Ast(expr)) => try_visit!(visitor.visit_expr(expr)),
        AttrArgs::Eq(_, AttrArgsEq::Hir(lit)) => {
            unreachable!("in literal form when walking mac args eq: {:?}", lit)
        }
    }
    V::Result::output()
}
