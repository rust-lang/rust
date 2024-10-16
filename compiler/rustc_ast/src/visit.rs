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

pub use rustc_ast_ir::visit::VisitorResult;
pub use rustc_ast_ir::{try_visit, visit_opt, walk_list, walk_visitable_list};
use rustc_span::Span;
use rustc_span::symbol::Ident;

use crate::ast::*;
use crate::ptr::P;

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
impl BoundKind {
    pub fn descr(self) -> &'static str {
        match self {
            BoundKind::Bound => "bounds",
            BoundKind::Impl => "`impl Trait`",
            BoundKind::TraitObject => "`dyn` trait object bounds",
            BoundKind::SuperTraits => "supertrait bounds",
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum FnKind<'a> {
    /// E.g., `fn foo()`, `fn foo(&self)`, or `extern "Abi" fn foo()`.
    Fn(FnCtxt, &'a Ident, &'a FnSig, &'a Visibility, &'a Generics, &'a Option<P<Block>>),

    /// E.g., `|x, y| body`.
    Closure(&'a ClosureBinder, &'a Option<CoroutineKind>, &'a FnDecl, &'a Expr),
}

impl<'a> FnKind<'a> {
    pub fn header(&self) -> Option<&'a FnHeader> {
        match *self {
            FnKind::Fn(_, _, sig, _, _, _) => Some(&sig.header),
            FnKind::Closure(..) => None,
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
            FnKind::Closure(_, _, decl, _) => decl,
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

pub trait WalkItemKind {
    type Ctxt;
    fn walk<'a, V: Visitor<'a>>(
        &'a self,
        span: Span,
        id: NodeId,
        ident: &'a Ident,
        visibility: &'a Visibility,
        ctxt: Self::Ctxt,
        visitor: &mut V,
    ) -> V::Result;
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

    fn visit_ident(&mut self, _ident: &'ast Ident) -> Self::Result {
        Self::Result::output()
    }
    fn visit_foreign_item(&mut self, i: &'ast ForeignItem) -> Self::Result {
        walk_item(self, i)
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
    fn visit_where_predicate_kind(&mut self, k: &'ast WherePredicateKind) -> Self::Result {
        walk_where_predicate_kind(self, k)
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
    fn visit_precise_capturing_arg(&mut self, arg: &'ast PreciseCapturingArg) -> Self::Result {
        walk_precise_capturing_arg(self, arg)
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
    fn visit_assoc_item_constraint(
        &mut self,
        constraint: &'ast AssocItemConstraint,
    ) -> Self::Result {
        walk_assoc_item_constraint(self, constraint)
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
    fn visit_fn_header(&mut self, header: &'ast FnHeader) -> Self::Result {
        walk_fn_header(self, header)
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
    fn visit_coroutine_kind(&mut self, _coroutine_kind: &'ast CoroutineKind) -> Self::Result {
        Self::Result::output()
    }
    fn visit_fn_decl(&mut self, fn_decl: &'ast FnDecl) -> Self::Result {
        walk_fn_decl(self, fn_decl)
    }
    fn visit_qself(&mut self, qs: &'ast Option<P<QSelf>>) -> Self::Result {
        walk_qself(self, qs)
    }
}

pub fn walk_crate<'a, V: Visitor<'a>>(visitor: &mut V, krate: &'a Crate) -> V::Result {
    let Crate { attrs, items, spans: _, id: _, is_placeholder: _ } = krate;
    walk_list!(visitor, visit_attribute, attrs);
    walk_list!(visitor, visit_item, items);
    V::Result::output()
}

pub fn walk_local<'a, V: Visitor<'a>>(visitor: &mut V, local: &'a Local) -> V::Result {
    let Local { id: _, pat, ty, kind, span: _, colon_sp: _, attrs, tokens: _ } = local;
    walk_list!(visitor, visit_attribute, attrs);
    try_visit!(visitor.visit_pat(pat));
    visit_opt!(visitor, visit_ty, ty);
    if let Some((init, els)) = kind.init_else_opt() {
        try_visit!(visitor.visit_expr(init));
        visit_opt!(visitor, visit_block, els);
    }
    V::Result::output()
}

pub fn walk_label<'a, V: Visitor<'a>>(visitor: &mut V, Label { ident }: &'a Label) -> V::Result {
    visitor.visit_ident(ident)
}

pub fn walk_lifetime<'a, V: Visitor<'a>>(visitor: &mut V, lifetime: &'a Lifetime) -> V::Result {
    let Lifetime { id: _, ident } = lifetime;
    visitor.visit_ident(ident)
}

pub fn walk_poly_trait_ref<'a, V>(visitor: &mut V, trait_ref: &'a PolyTraitRef) -> V::Result
where
    V: Visitor<'a>,
{
    let PolyTraitRef { bound_generic_params, modifiers: _, trait_ref, span: _ } = trait_ref;
    walk_list!(visitor, visit_generic_param, bound_generic_params);
    visitor.visit_trait_ref(trait_ref)
}

pub fn walk_trait_ref<'a, V: Visitor<'a>>(visitor: &mut V, trait_ref: &'a TraitRef) -> V::Result {
    let TraitRef { path, ref_id } = trait_ref;
    visitor.visit_path(path, *ref_id)
}

impl WalkItemKind for ItemKind {
    type Ctxt = ();
    fn walk<'a, V: Visitor<'a>>(
        &'a self,
        span: Span,
        id: NodeId,
        ident: &'a Ident,
        vis: &'a Visibility,
        _ctxt: Self::Ctxt,
        visitor: &mut V,
    ) -> V::Result {
        match self {
            ItemKind::ExternCrate(_rename) => {}
            ItemKind::Use(use_tree) => try_visit!(visitor.visit_use_tree(use_tree, id, false)),
            ItemKind::Static(box StaticItem { ty, safety: _, mutability: _, expr }) => {
                try_visit!(visitor.visit_ty(ty));
                visit_opt!(visitor, visit_expr, expr);
            }
            ItemKind::Const(box ConstItem { defaultness: _, generics, ty, expr }) => {
                try_visit!(visitor.visit_generics(generics));
                try_visit!(visitor.visit_ty(ty));
                visit_opt!(visitor, visit_expr, expr);
            }
            ItemKind::Fn(box Fn { defaultness: _, generics, sig, body }) => {
                let kind = FnKind::Fn(FnCtxt::Free, ident, sig, vis, generics, body);
                try_visit!(visitor.visit_fn(kind, span, id));
            }
            ItemKind::Mod(_unsafety, mod_kind) => match mod_kind {
                ModKind::Loaded(items, _inline, _inner_span) => {
                    walk_list!(visitor, visit_item, items);
                }
                ModKind::Unloaded => {}
            },
            ItemKind::ForeignMod(ForeignMod { extern_span: _, safety: _, abi: _, items }) => {
                walk_list!(visitor, visit_foreign_item, items);
            }
            ItemKind::GlobalAsm(asm) => try_visit!(visitor.visit_inline_asm(asm)),
            ItemKind::TyAlias(box TyAlias {
                generics,
                bounds,
                ty,
                defaultness: _,
                where_clauses: _,
            }) => {
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
                safety: _,
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
            ItemKind::Trait(box Trait { safety: _, is_auto: _, generics, bounds, items }) => {
                try_visit!(visitor.visit_generics(generics));
                walk_list!(visitor, visit_param_bound, bounds, BoundKind::SuperTraits);
                walk_list!(visitor, visit_assoc_item, items, AssocCtxt::Trait);
            }
            ItemKind::TraitAlias(generics, bounds) => {
                try_visit!(visitor.visit_generics(generics));
                walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
            }
            ItemKind::MacCall(mac) => try_visit!(visitor.visit_mac_call(mac)),
            ItemKind::MacroDef(ts) => try_visit!(visitor.visit_mac_def(ts, id)),
            ItemKind::Delegation(box Delegation {
                id,
                qself,
                path,
                rename,
                body,
                from_glob: _,
            }) => {
                try_visit!(visitor.visit_qself(qself));
                try_visit!(visitor.visit_path(path, *id));
                visit_opt!(visitor, visit_ident, rename);
                visit_opt!(visitor, visit_block, body);
            }
            ItemKind::DelegationMac(box DelegationMac { qself, prefix, suffixes, body }) => {
                try_visit!(visitor.visit_qself(qself));
                try_visit!(visitor.visit_path(prefix, id));
                if let Some(suffixes) = suffixes {
                    for (ident, rename) in suffixes {
                        visitor.visit_ident(ident);
                        if let Some(rename) = rename {
                            visitor.visit_ident(rename);
                        }
                    }
                }
                visit_opt!(visitor, visit_block, body);
            }
        }
        V::Result::output()
    }
}

pub fn walk_enum_def<'a, V: Visitor<'a>>(
    visitor: &mut V,
    EnumDef { variants }: &'a EnumDef,
) -> V::Result {
    walk_list!(visitor, visit_variant, variants);
    V::Result::output()
}

pub fn walk_variant<'a, V: Visitor<'a>>(visitor: &mut V, variant: &'a Variant) -> V::Result
where
    V: Visitor<'a>,
{
    let Variant { attrs, id: _, span: _, vis, ident, data, disr_expr, is_placeholder: _ } = variant;
    walk_list!(visitor, visit_attribute, attrs);
    try_visit!(visitor.visit_vis(vis));
    try_visit!(visitor.visit_ident(ident));
    try_visit!(visitor.visit_variant_data(data));
    visit_opt!(visitor, visit_variant_discr, disr_expr);
    V::Result::output()
}

pub fn walk_expr_field<'a, V: Visitor<'a>>(visitor: &mut V, f: &'a ExprField) -> V::Result {
    let ExprField { attrs, id: _, span: _, ident, expr, is_shorthand: _, is_placeholder: _ } = f;
    walk_list!(visitor, visit_attribute, attrs);
    try_visit!(visitor.visit_ident(ident));
    try_visit!(visitor.visit_expr(expr));
    V::Result::output()
}

pub fn walk_pat_field<'a, V: Visitor<'a>>(visitor: &mut V, fp: &'a PatField) -> V::Result {
    let PatField { ident, pat, is_shorthand: _, attrs, id: _, span: _, is_placeholder: _ } = fp;
    walk_list!(visitor, visit_attribute, attrs);
    try_visit!(visitor.visit_ident(ident));
    try_visit!(visitor.visit_pat(pat));
    V::Result::output()
}

pub fn walk_ty<'a, V: Visitor<'a>>(visitor: &mut V, typ: &'a Ty) -> V::Result {
    let Ty { id, kind, span: _, tokens: _ } = typ;
    match kind {
        TyKind::Slice(ty) | TyKind::Paren(ty) => try_visit!(visitor.visit_ty(ty)),
        TyKind::Ptr(MutTy { ty, mutbl: _ }) => try_visit!(visitor.visit_ty(ty)),
        TyKind::Ref(opt_lifetime, MutTy { ty, mutbl: _ })
        | TyKind::PinnedRef(opt_lifetime, MutTy { ty, mutbl: _ }) => {
            visit_opt!(visitor, visit_lifetime, opt_lifetime, LifetimeCtxt::Ref);
            try_visit!(visitor.visit_ty(ty));
        }
        TyKind::Tup(tuple_element_types) => {
            walk_list!(visitor, visit_ty, tuple_element_types);
        }
        TyKind::BareFn(function_declaration) => {
            let BareFnTy { safety: _, ext: _, generic_params, decl, decl_span: _ } =
                &**function_declaration;
            walk_list!(visitor, visit_generic_param, generic_params);
            try_visit!(visitor.visit_fn_decl(decl));
        }
        TyKind::Path(maybe_qself, path) => {
            try_visit!(visitor.visit_qself(maybe_qself));
            try_visit!(visitor.visit_path(path, *id));
        }
        TyKind::Pat(ty, pat) => {
            try_visit!(visitor.visit_ty(ty));
            try_visit!(visitor.visit_pat(pat));
        }
        TyKind::Array(ty, length) => {
            try_visit!(visitor.visit_ty(ty));
            try_visit!(visitor.visit_anon_const(length));
        }
        TyKind::TraitObject(bounds, _syntax) => {
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::TraitObject);
        }
        TyKind::ImplTrait(_id, bounds) => {
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::Impl);
        }
        TyKind::Typeof(expression) => try_visit!(visitor.visit_anon_const(expression)),
        TyKind::Infer | TyKind::ImplicitSelf | TyKind::Dummy => {}
        TyKind::Err(_guar) => {}
        TyKind::MacCall(mac) => try_visit!(visitor.visit_mac_call(mac)),
        TyKind::Never | TyKind::CVarArgs => {}
    }
    V::Result::output()
}

fn walk_qself<'a, V: Visitor<'a>>(visitor: &mut V, qself: &'a Option<P<QSelf>>) -> V::Result {
    if let Some(qself) = qself {
        let QSelf { ty, path_span: _, position: _ } = &**qself;
        try_visit!(visitor.visit_ty(ty));
    }
    V::Result::output()
}

pub fn walk_path<'a, V: Visitor<'a>>(visitor: &mut V, path: &'a Path) -> V::Result {
    let Path { span: _, segments, tokens: _ } = path;
    walk_list!(visitor, visit_path_segment, segments);
    V::Result::output()
}

pub fn walk_use_tree<'a, V: Visitor<'a>>(
    visitor: &mut V,
    use_tree: &'a UseTree,
    id: NodeId,
) -> V::Result {
    let UseTree { prefix, kind, span: _ } = use_tree;
    try_visit!(visitor.visit_path(prefix, id));
    match kind {
        UseTreeKind::Simple(rename) => {
            // The extra IDs are handled during AST lowering.
            visit_opt!(visitor, visit_ident, rename);
        }
        UseTreeKind::Glob => {}
        UseTreeKind::Nested { ref items, span: _ } => {
            for &(ref nested_tree, nested_id) in items {
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
    let PathSegment { ident, id: _, args } = segment;
    try_visit!(visitor.visit_ident(ident));
    visit_opt!(visitor, visit_generic_args, args);
    V::Result::output()
}

pub fn walk_generic_args<'a, V>(visitor: &mut V, generic_args: &'a GenericArgs) -> V::Result
where
    V: Visitor<'a>,
{
    match generic_args {
        GenericArgs::AngleBracketed(AngleBracketedArgs { span: _, args }) => {
            for arg in args {
                match arg {
                    AngleBracketedArg::Arg(a) => try_visit!(visitor.visit_generic_arg(a)),
                    AngleBracketedArg::Constraint(c) => {
                        try_visit!(visitor.visit_assoc_item_constraint(c))
                    }
                }
            }
        }
        GenericArgs::Parenthesized(data) => {
            let ParenthesizedArgs { span: _, inputs, inputs_span: _, output } = data;
            walk_list!(visitor, visit_ty, inputs);
            try_visit!(visitor.visit_fn_ret_ty(output));
        }
        GenericArgs::ParenthesizedElided(_span) => {}
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

pub fn walk_assoc_item_constraint<'a, V: Visitor<'a>>(
    visitor: &mut V,
    constraint: &'a AssocItemConstraint,
) -> V::Result {
    let AssocItemConstraint { id: _, ident, gen_args, kind, span: _ } = constraint;
    try_visit!(visitor.visit_ident(ident));
    visit_opt!(visitor, visit_generic_args, gen_args);
    match kind {
        AssocItemConstraintKind::Equality { term } => match term {
            Term::Ty(ty) => try_visit!(visitor.visit_ty(ty)),
            Term::Const(c) => try_visit!(visitor.visit_anon_const(c)),
        },
        AssocItemConstraintKind::Bound { bounds } => {
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
        }
    }
    V::Result::output()
}

pub fn walk_pat<'a, V: Visitor<'a>>(visitor: &mut V, pattern: &'a Pat) -> V::Result {
    let Pat { id, kind, span: _, tokens: _ } = pattern;
    match kind {
        PatKind::TupleStruct(opt_qself, path, elems) => {
            try_visit!(visitor.visit_qself(opt_qself));
            try_visit!(visitor.visit_path(path, *id));
            walk_list!(visitor, visit_pat, elems);
        }
        PatKind::Path(opt_qself, path) => {
            try_visit!(visitor.visit_qself(opt_qself));
            try_visit!(visitor.visit_path(path, *id))
        }
        PatKind::Struct(opt_qself, path, fields, _rest) => {
            try_visit!(visitor.visit_qself(opt_qself));
            try_visit!(visitor.visit_path(path, *id));
            walk_list!(visitor, visit_pat_field, fields);
        }
        PatKind::Box(subpattern) | PatKind::Deref(subpattern) | PatKind::Paren(subpattern) => {
            try_visit!(visitor.visit_pat(subpattern));
        }
        PatKind::Ref(subpattern, _ /*mutbl*/) => {
            try_visit!(visitor.visit_pat(subpattern));
        }
        PatKind::Ident(_bmode, ident, optional_subpattern) => {
            try_visit!(visitor.visit_ident(ident));
            visit_opt!(visitor, visit_pat, optional_subpattern);
        }
        PatKind::Lit(expression) => try_visit!(visitor.visit_expr(expression)),
        PatKind::Range(lower_bound, upper_bound, _end) => {
            visit_opt!(visitor, visit_expr, lower_bound);
            visit_opt!(visitor, visit_expr, upper_bound);
        }
        PatKind::Wild | PatKind::Rest | PatKind::Never => {}
        PatKind::Err(_guar) => {}
        PatKind::Tuple(elems) | PatKind::Slice(elems) | PatKind::Or(elems) => {
            walk_list!(visitor, visit_pat, elems);
        }
        PatKind::MacCall(mac) => try_visit!(visitor.visit_mac_call(mac)),
    }
    V::Result::output()
}

impl WalkItemKind for ForeignItemKind {
    type Ctxt = ();
    fn walk<'a, V: Visitor<'a>>(
        &'a self,
        span: Span,
        id: NodeId,
        ident: &'a Ident,
        vis: &'a Visibility,
        _ctxt: Self::Ctxt,
        visitor: &mut V,
    ) -> V::Result {
        match self {
            ForeignItemKind::Static(box StaticItem { ty, mutability: _, expr, safety: _ }) => {
                try_visit!(visitor.visit_ty(ty));
                visit_opt!(visitor, visit_expr, expr);
            }
            ForeignItemKind::Fn(box Fn { defaultness: _, generics, sig, body }) => {
                let kind = FnKind::Fn(FnCtxt::Foreign, ident, sig, vis, generics, body);
                try_visit!(visitor.visit_fn(kind, span, id));
            }
            ForeignItemKind::TyAlias(box TyAlias {
                generics,
                bounds,
                ty,
                defaultness: _,
                where_clauses: _,
            }) => {
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
}

pub fn walk_param_bound<'a, V: Visitor<'a>>(visitor: &mut V, bound: &'a GenericBound) -> V::Result {
    match bound {
        GenericBound::Trait(trait_ref) => visitor.visit_poly_trait_ref(trait_ref),
        GenericBound::Outlives(lifetime) => visitor.visit_lifetime(lifetime, LifetimeCtxt::Bound),
        GenericBound::Use(args, _span) => {
            walk_list!(visitor, visit_precise_capturing_arg, args);
            V::Result::output()
        }
    }
}

pub fn walk_precise_capturing_arg<'a, V: Visitor<'a>>(
    visitor: &mut V,
    arg: &'a PreciseCapturingArg,
) -> V::Result {
    match arg {
        PreciseCapturingArg::Lifetime(lt) => visitor.visit_lifetime(lt, LifetimeCtxt::GenericArg),
        PreciseCapturingArg::Arg(path, id) => visitor.visit_path(path, *id),
    }
}

pub fn walk_generic_param<'a, V: Visitor<'a>>(
    visitor: &mut V,
    param: &'a GenericParam,
) -> V::Result {
    let GenericParam { id: _, ident, attrs, bounds, is_placeholder: _, kind, colon_span: _ } =
        param;
    walk_list!(visitor, visit_attribute, attrs);
    try_visit!(visitor.visit_ident(ident));
    walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
    match kind {
        GenericParamKind::Lifetime => (),
        GenericParamKind::Type { default } => visit_opt!(visitor, visit_ty, default),
        GenericParamKind::Const { ty, default, kw_span: _ } => {
            try_visit!(visitor.visit_ty(ty));
            visit_opt!(visitor, visit_anon_const, default);
        }
    }
    V::Result::output()
}

pub fn walk_generics<'a, V: Visitor<'a>>(visitor: &mut V, generics: &'a Generics) -> V::Result {
    let Generics { params, where_clause, span: _ } = generics;
    let WhereClause { has_where_token: _, predicates, span: _ } = where_clause;
    walk_list!(visitor, visit_generic_param, params);
    walk_list!(visitor, visit_where_predicate, predicates);
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
    let WherePredicate { kind, id: _, span: _ } = predicate;
    visitor.visit_where_predicate_kind(kind)
}

pub fn walk_where_predicate_kind<'a, V: Visitor<'a>>(
    visitor: &mut V,
    kind: &'a WherePredicateKind,
) -> V::Result {
    match kind {
        WherePredicateKind::BoundPredicate(WhereBoundPredicate {
            bounded_ty,
            bounds,
            bound_generic_params,
        }) => {
            walk_list!(visitor, visit_generic_param, bound_generic_params);
            try_visit!(visitor.visit_ty(bounded_ty));
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
        }
        WherePredicateKind::RegionPredicate(WhereRegionPredicate { lifetime, bounds }) => {
            try_visit!(visitor.visit_lifetime(lifetime, LifetimeCtxt::Bound));
            walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
        }
        WherePredicateKind::EqPredicate(WhereEqPredicate { lhs_ty, rhs_ty }) => {
            try_visit!(visitor.visit_ty(lhs_ty));
            try_visit!(visitor.visit_ty(rhs_ty));
        }
    }
    V::Result::output()
}

pub fn walk_fn_ret_ty<'a, V: Visitor<'a>>(visitor: &mut V, ret_ty: &'a FnRetTy) -> V::Result {
    match ret_ty {
        FnRetTy::Default(_span) => {}
        FnRetTy::Ty(output_ty) => try_visit!(visitor.visit_ty(output_ty)),
    }
    V::Result::output()
}

pub fn walk_fn_header<'a, V: Visitor<'a>>(visitor: &mut V, fn_header: &'a FnHeader) -> V::Result {
    let FnHeader { safety: _, coroutine_kind, constness: _, ext: _ } = fn_header;
    visit_opt!(visitor, visit_coroutine_kind, coroutine_kind.as_ref());
    V::Result::output()
}

pub fn walk_fn_decl<'a, V: Visitor<'a>>(
    visitor: &mut V,
    FnDecl { inputs, output }: &'a FnDecl,
) -> V::Result {
    walk_list!(visitor, visit_param, inputs);
    visitor.visit_fn_ret_ty(output)
}

pub fn walk_fn<'a, V: Visitor<'a>>(visitor: &mut V, kind: FnKind<'a>) -> V::Result {
    match kind {
        FnKind::Fn(_ctxt, _ident, FnSig { header, decl, span: _ }, _vis, generics, body) => {
            // Identifier and visibility are visited as a part of the item.
            try_visit!(visitor.visit_fn_header(header));
            try_visit!(visitor.visit_generics(generics));
            try_visit!(visitor.visit_fn_decl(decl));
            visit_opt!(visitor, visit_block, body);
        }
        FnKind::Closure(binder, coroutine_kind, decl, body) => {
            try_visit!(visitor.visit_closure_binder(binder));
            visit_opt!(visitor, visit_coroutine_kind, coroutine_kind.as_ref());
            try_visit!(visitor.visit_fn_decl(decl));
            try_visit!(visitor.visit_expr(body));
        }
    }
    V::Result::output()
}

impl WalkItemKind for AssocItemKind {
    type Ctxt = AssocCtxt;
    fn walk<'a, V: Visitor<'a>>(
        &'a self,
        span: Span,
        id: NodeId,
        ident: &'a Ident,
        vis: &'a Visibility,
        ctxt: Self::Ctxt,
        visitor: &mut V,
    ) -> V::Result {
        match self {
            AssocItemKind::Const(box ConstItem { defaultness: _, generics, ty, expr }) => {
                try_visit!(visitor.visit_generics(generics));
                try_visit!(visitor.visit_ty(ty));
                visit_opt!(visitor, visit_expr, expr);
            }
            AssocItemKind::Fn(box Fn { defaultness: _, generics, sig, body }) => {
                let kind = FnKind::Fn(FnCtxt::Assoc(ctxt), ident, sig, vis, generics, body);
                try_visit!(visitor.visit_fn(kind, span, id));
            }
            AssocItemKind::Type(box TyAlias {
                generics,
                bounds,
                ty,
                defaultness: _,
                where_clauses: _,
            }) => {
                try_visit!(visitor.visit_generics(generics));
                walk_list!(visitor, visit_param_bound, bounds, BoundKind::Bound);
                visit_opt!(visitor, visit_ty, ty);
            }
            AssocItemKind::MacCall(mac) => {
                try_visit!(visitor.visit_mac_call(mac));
            }
            AssocItemKind::Delegation(box Delegation {
                id,
                qself,
                path,
                rename,
                body,
                from_glob: _,
            }) => {
                try_visit!(visitor.visit_qself(qself));
                try_visit!(visitor.visit_path(path, *id));
                visit_opt!(visitor, visit_ident, rename);
                visit_opt!(visitor, visit_block, body);
            }
            AssocItemKind::DelegationMac(box DelegationMac { qself, prefix, suffixes, body }) => {
                try_visit!(visitor.visit_qself(qself));
                try_visit!(visitor.visit_path(prefix, id));
                if let Some(suffixes) = suffixes {
                    for (ident, rename) in suffixes {
                        visitor.visit_ident(ident);
                        if let Some(rename) = rename {
                            visitor.visit_ident(rename);
                        }
                    }
                }
                visit_opt!(visitor, visit_block, body);
            }
        }
        V::Result::output()
    }
}

pub fn walk_item<'a, V: Visitor<'a>>(
    visitor: &mut V,
    item: &'a Item<impl WalkItemKind<Ctxt = ()>>,
) -> V::Result {
    walk_item_ctxt(visitor, item, ())
}

pub fn walk_assoc_item<'a, V: Visitor<'a>>(
    visitor: &mut V,
    item: &'a AssocItem,
    ctxt: AssocCtxt,
) -> V::Result {
    walk_item_ctxt(visitor, item, ctxt)
}

fn walk_item_ctxt<'a, V: Visitor<'a>, K: WalkItemKind>(
    visitor: &mut V,
    item: &'a Item<K>,
    ctxt: K::Ctxt,
) -> V::Result {
    let Item { id, span, ident, vis, attrs, kind, tokens: _ } = item;
    walk_list!(visitor, visit_attribute, attrs);
    try_visit!(visitor.visit_vis(vis));
    try_visit!(visitor.visit_ident(ident));
    try_visit!(kind.walk(*span, *id, ident, vis, ctxt, visitor));
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
    let FieldDef { attrs, id: _, span: _, vis, ident, ty, is_placeholder: _, safety: _ } = field;
    walk_list!(visitor, visit_attribute, attrs);
    try_visit!(visitor.visit_vis(vis));
    visit_opt!(visitor, visit_ident, ident);
    try_visit!(visitor.visit_ty(ty));
    V::Result::output()
}

pub fn walk_block<'a, V: Visitor<'a>>(visitor: &mut V, block: &'a Block) -> V::Result {
    let Block { stmts, id: _, rules: _, span: _, tokens: _, could_be_bare_literal: _ } = block;
    walk_list!(visitor, visit_stmt, stmts);
    V::Result::output()
}

pub fn walk_stmt<'a, V: Visitor<'a>>(visitor: &mut V, statement: &'a Stmt) -> V::Result {
    let Stmt { id: _, kind, span: _ } = statement;
    match kind {
        StmtKind::Let(local) => try_visit!(visitor.visit_local(local)),
        StmtKind::Item(item) => try_visit!(visitor.visit_item(item)),
        StmtKind::Expr(expr) | StmtKind::Semi(expr) => try_visit!(visitor.visit_expr(expr)),
        StmtKind::Empty => {}
        StmtKind::MacCall(mac) => {
            let MacCallStmt { mac, attrs, style: _, tokens: _ } = &**mac;
            walk_list!(visitor, visit_attribute, attrs);
            try_visit!(visitor.visit_mac_call(mac));
        }
    }
    V::Result::output()
}

pub fn walk_mac<'a, V: Visitor<'a>>(visitor: &mut V, mac: &'a MacCall) -> V::Result {
    let MacCall { path, args: _ } = mac;
    visitor.visit_path(path, DUMMY_NODE_ID)
}

pub fn walk_anon_const<'a, V: Visitor<'a>>(visitor: &mut V, constant: &'a AnonConst) -> V::Result {
    let AnonConst { id: _, value } = constant;
    visitor.visit_expr(value)
}

pub fn walk_inline_asm<'a, V: Visitor<'a>>(visitor: &mut V, asm: &'a InlineAsm) -> V::Result {
    let InlineAsm {
        asm_macro: _,
        template: _,
        template_strs: _,
        operands,
        clobber_abis: _,
        options: _,
        line_spans: _,
    } = asm;
    for (op, _span) in operands {
        match op {
            InlineAsmOperand::In { expr, reg: _ }
            | InlineAsmOperand::Out { expr: Some(expr), reg: _, late: _ }
            | InlineAsmOperand::InOut { expr, reg: _, late: _ } => {
                try_visit!(visitor.visit_expr(expr))
            }
            InlineAsmOperand::Out { expr: None, reg: _, late: _ } => {}
            InlineAsmOperand::SplitInOut { in_expr, out_expr, reg: _, late: _ } => {
                try_visit!(visitor.visit_expr(in_expr));
                visit_opt!(visitor, visit_expr, out_expr);
            }
            InlineAsmOperand::Const { anon_const } => {
                try_visit!(visitor.visit_anon_const(anon_const))
            }
            InlineAsmOperand::Sym { sym } => try_visit!(visitor.visit_inline_asm_sym(sym)),
            InlineAsmOperand::Label { block } => try_visit!(visitor.visit_block(block)),
        }
    }
    V::Result::output()
}

pub fn walk_inline_asm_sym<'a, V: Visitor<'a>>(
    visitor: &mut V,
    InlineAsmSym { id, qself, path }: &'a InlineAsmSym,
) -> V::Result {
    try_visit!(visitor.visit_qself(qself));
    visitor.visit_path(path, *id)
}

pub fn walk_format_args<'a, V: Visitor<'a>>(visitor: &mut V, fmt: &'a FormatArgs) -> V::Result {
    let FormatArgs { span: _, template: _, arguments } = fmt;
    for FormatArgument { kind, expr } in arguments.all_args() {
        match kind {
            FormatArgumentKind::Named(ident) | FormatArgumentKind::Captured(ident) => {
                try_visit!(visitor.visit_ident(ident))
            }
            FormatArgumentKind::Normal => {}
        }
        try_visit!(visitor.visit_expr(expr));
    }
    V::Result::output()
}

pub fn walk_expr<'a, V: Visitor<'a>>(visitor: &mut V, expression: &'a Expr) -> V::Result {
    let Expr { id, kind, span, attrs, tokens: _ } = expression;
    walk_list!(visitor, visit_attribute, attrs);
    match kind {
        ExprKind::Array(subexpressions) => {
            walk_list!(visitor, visit_expr, subexpressions);
        }
        ExprKind::ConstBlock(anon_const) => try_visit!(visitor.visit_anon_const(anon_const)),
        ExprKind::Repeat(element, count) => {
            try_visit!(visitor.visit_expr(element));
            try_visit!(visitor.visit_anon_const(count));
        }
        ExprKind::Struct(se) => {
            let StructExpr { qself, path, fields, rest } = &**se;
            try_visit!(visitor.visit_qself(qself));
            try_visit!(visitor.visit_path(path, *id));
            walk_list!(visitor, visit_expr_field, fields);
            match rest {
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
            try_visit!(visitor.visit_expr(receiver));
            try_visit!(visitor.visit_path_segment(seg));
            walk_list!(visitor, visit_expr, args);
        }
        ExprKind::Binary(_op, left_expression, right_expression) => {
            try_visit!(visitor.visit_expr(left_expression));
            try_visit!(visitor.visit_expr(right_expression));
        }
        ExprKind::AddrOf(_kind, _mutbl, subexpression) => {
            try_visit!(visitor.visit_expr(subexpression));
        }
        ExprKind::Unary(_op, subexpression) => {
            try_visit!(visitor.visit_expr(subexpression));
        }
        ExprKind::Cast(subexpression, typ) | ExprKind::Type(subexpression, typ) => {
            try_visit!(visitor.visit_expr(subexpression));
            try_visit!(visitor.visit_ty(typ));
        }
        ExprKind::Let(pat, expr, _span, _recovered) => {
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
        ExprKind::Loop(block, opt_label, _span) => {
            visit_opt!(visitor, visit_label, opt_label);
            try_visit!(visitor.visit_block(block));
        }
        ExprKind::Match(subexpression, arms, _kind) => {
            try_visit!(visitor.visit_expr(subexpression));
            walk_list!(visitor, visit_arm, arms);
        }
        ExprKind::Closure(box Closure {
            binder,
            capture_clause,
            coroutine_kind,
            constness: _,
            movability: _,
            fn_decl,
            body,
            fn_decl_span: _,
            fn_arg_span: _,
        }) => {
            try_visit!(visitor.visit_capture_by(capture_clause));
            try_visit!(visitor.visit_fn(
                FnKind::Closure(binder, coroutine_kind, fn_decl, body),
                *span,
                *id
            ))
        }
        ExprKind::Block(block, opt_label) => {
            visit_opt!(visitor, visit_label, opt_label);
            try_visit!(visitor.visit_block(block));
        }
        ExprKind::Gen(_capt, body, _kind, _decl_span) => try_visit!(visitor.visit_block(body)),
        ExprKind::Await(expr, _span) => try_visit!(visitor.visit_expr(expr)),
        ExprKind::Assign(lhs, rhs, _span) => {
            try_visit!(visitor.visit_expr(lhs));
            try_visit!(visitor.visit_expr(rhs));
        }
        ExprKind::AssignOp(_op, left_expression, right_expression) => {
            try_visit!(visitor.visit_expr(left_expression));
            try_visit!(visitor.visit_expr(right_expression));
        }
        ExprKind::Field(subexpression, ident) => {
            try_visit!(visitor.visit_expr(subexpression));
            try_visit!(visitor.visit_ident(ident));
        }
        ExprKind::Index(main_expression, index_expression, _span) => {
            try_visit!(visitor.visit_expr(main_expression));
            try_visit!(visitor.visit_expr(index_expression));
        }
        ExprKind::Range(start, end, _limit) => {
            visit_opt!(visitor, visit_expr, start);
            visit_opt!(visitor, visit_expr, end);
        }
        ExprKind::Underscore => {}
        ExprKind::Path(maybe_qself, path) => {
            try_visit!(visitor.visit_qself(maybe_qself));
            try_visit!(visitor.visit_path(path, *id));
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
            try_visit!(visitor.visit_ty(container));
            walk_list!(visitor, visit_ident, fields.iter());
        }
        ExprKind::Yield(optional_expression) => {
            visit_opt!(visitor, visit_expr, optional_expression);
        }
        ExprKind::Try(subexpression) => try_visit!(visitor.visit_expr(subexpression)),
        ExprKind::TryBlock(body) => try_visit!(visitor.visit_block(body)),
        ExprKind::Lit(_token) => {}
        ExprKind::IncludedBytes(_bytes) => {}
        ExprKind::Err(_guar) => {}
        ExprKind::Dummy => {}
    }

    V::Result::output()
}

pub fn walk_param<'a, V: Visitor<'a>>(visitor: &mut V, param: &'a Param) -> V::Result {
    let Param { attrs, ty, pat, id: _, span: _, is_placeholder: _ } = param;
    walk_list!(visitor, visit_attribute, attrs);
    try_visit!(visitor.visit_pat(pat));
    try_visit!(visitor.visit_ty(ty));
    V::Result::output()
}

pub fn walk_arm<'a, V: Visitor<'a>>(visitor: &mut V, arm: &'a Arm) -> V::Result {
    let Arm { attrs, pat, guard, body, span: _, id: _, is_placeholder: _ } = arm;
    walk_list!(visitor, visit_attribute, attrs);
    try_visit!(visitor.visit_pat(pat));
    visit_opt!(visitor, visit_expr, guard);
    visit_opt!(visitor, visit_expr, body);
    V::Result::output()
}

pub fn walk_vis<'a, V: Visitor<'a>>(visitor: &mut V, vis: &'a Visibility) -> V::Result {
    let Visibility { kind, span: _, tokens: _ } = vis;
    match kind {
        VisibilityKind::Restricted { path, id, shorthand: _ } => {
            try_visit!(visitor.visit_path(path, *id));
        }
        VisibilityKind::Public | VisibilityKind::Inherited => {}
    }
    V::Result::output()
}

pub fn walk_attribute<'a, V: Visitor<'a>>(visitor: &mut V, attr: &'a Attribute) -> V::Result {
    let Attribute { kind, id: _, style: _, span: _ } = attr;
    match kind {
        AttrKind::Normal(normal) => {
            let NormalAttr { item, tokens: _ } = &**normal;
            let AttrItem { unsafety: _, path, args, tokens: _ } = item;
            try_visit!(visitor.visit_path(path, DUMMY_NODE_ID));
            try_visit!(walk_attr_args(visitor, args));
        }
        AttrKind::DocComment(_kind, _sym) => {}
    }
    V::Result::output()
}

pub fn walk_attr_args<'a, V: Visitor<'a>>(visitor: &mut V, args: &'a AttrArgs) -> V::Result {
    match args {
        AttrArgs::Empty => {}
        AttrArgs::Delimited(_args) => {}
        AttrArgs::Eq { expr, .. } => try_visit!(visitor.visit_expr(expr)),
    }
    V::Result::output()
}
