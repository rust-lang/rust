//! A `MutVisitor` represents an AST modification; it accepts an AST piece and
//! mutates it in place. So, for instance, macro expansion is a `MutVisitor`
//! that walks over an AST and modifies it.
//!
//! Note: using a `MutVisitor` (other than the `MacroExpander` `MutVisitor`) on
//! an AST before macro expansion is probably a bad idea. For instance,
//! a `MutVisitor` renaming item names in a module will miss all of those
//! that are created by the expansion of a macro.

use crate::ast::*;
use crate::ptr::P;
use crate::token::{self, Token};
use crate::tokenstream::*;

use rustc_data_structures::flat_map_in_place::FlatMapInPlace;
use rustc_data_structures::sync::Lrc;
use rustc_span::source_map::Spanned;
use rustc_span::symbol::Ident;
use rustc_span::Span;
use smallvec::{smallvec, Array, SmallVec};
use std::ops::DerefMut;
use std::{panic, ptr};
use thin_vec::ThinVec;

pub trait ExpectOne<A: Array> {
    fn expect_one(self, err: &'static str) -> A::Item;
}

impl<A: Array> ExpectOne<A> for SmallVec<A> {
    fn expect_one(self, err: &'static str) -> A::Item {
        assert!(self.len() == 1, "{}", err);
        self.into_iter().next().unwrap()
    }
}

pub trait MutVisitor: Sized {
    /// Mutable token visiting only exists for the `macro_rules` token marker and should not be
    /// used otherwise. Token visitor would be entirely separate from the regular visitor if
    /// the marker didn't have to visit AST fragments in nonterminal tokens.
    const VISIT_TOKENS: bool = false;

    // Methods in this trait have one of three forms:
    //
    //   fn visit_t(&mut self, t: &mut T);                      // common
    //   fn flat_map_t(&mut self, t: T) -> SmallVec<[T; 1]>;    // rare
    //   fn filter_map_t(&mut self, t: T) -> Option<T>;         // rarest
    //
    // Any additions to this trait should happen in form of a call to a public
    // `noop_*` function that only calls out to the visitor again, not other
    // `noop_*` functions. This is a necessary API workaround to the problem of
    // not being able to call out to the super default method in an overridden
    // default method.
    //
    // When writing these methods, it is better to use destructuring like this:
    //
    //   fn visit_abc(&mut self, ABC { a, b, c: _ }: &mut ABC) {
    //       visit_a(a);
    //       visit_b(b);
    //   }
    //
    // than to use field access like this:
    //
    //   fn visit_abc(&mut self, abc: &mut ABC) {
    //       visit_a(&mut abc.a);
    //       visit_b(&mut abc.b);
    //       // ignore abc.c
    //   }
    //
    // As well as being more concise, the former is explicit about which fields
    // are skipped. Furthermore, if a new field is added, the destructuring
    // version will cause a compile error, which is good. In comparison, the
    // field access version will continue working and it would be easy to
    // forget to add handling for it.

    fn visit_crate(&mut self, c: &mut Crate) {
        noop_visit_crate(c, self)
    }

    fn visit_meta_list_item(&mut self, list_item: &mut NestedMetaItem) {
        noop_visit_meta_list_item(list_item, self);
    }

    fn visit_meta_item(&mut self, meta_item: &mut MetaItem) {
        noop_visit_meta_item(meta_item, self);
    }

    fn visit_use_tree(&mut self, use_tree: &mut UseTree) {
        noop_visit_use_tree(use_tree, self);
    }

    fn flat_map_foreign_item(&mut self, ni: P<ForeignItem>) -> SmallVec<[P<ForeignItem>; 1]> {
        noop_flat_map_foreign_item(ni, self)
    }

    fn flat_map_item(&mut self, i: P<Item>) -> SmallVec<[P<Item>; 1]> {
        noop_flat_map_item(i, self)
    }

    fn visit_fn_header(&mut self, header: &mut FnHeader) {
        noop_visit_fn_header(header, self);
    }

    fn flat_map_field_def(&mut self, fd: FieldDef) -> SmallVec<[FieldDef; 1]> {
        noop_flat_map_field_def(fd, self)
    }

    fn visit_item_kind(&mut self, i: &mut ItemKind) {
        noop_visit_item_kind(i, self);
    }

    fn flat_map_trait_item(&mut self, i: P<AssocItem>) -> SmallVec<[P<AssocItem>; 1]> {
        noop_flat_map_assoc_item(i, self)
    }

    fn flat_map_impl_item(&mut self, i: P<AssocItem>) -> SmallVec<[P<AssocItem>; 1]> {
        noop_flat_map_assoc_item(i, self)
    }

    fn visit_fn_decl(&mut self, d: &mut P<FnDecl>) {
        noop_visit_fn_decl(d, self);
    }

    fn visit_asyncness(&mut self, a: &mut Async) {
        noop_visit_asyncness(a, self);
    }

    fn visit_closure_binder(&mut self, b: &mut ClosureBinder) {
        noop_visit_closure_binder(b, self);
    }

    fn visit_block(&mut self, b: &mut P<Block>) {
        noop_visit_block(b, self);
    }

    fn flat_map_stmt(&mut self, s: Stmt) -> SmallVec<[Stmt; 1]> {
        noop_flat_map_stmt(s, self)
    }

    fn flat_map_arm(&mut self, arm: Arm) -> SmallVec<[Arm; 1]> {
        noop_flat_map_arm(arm, self)
    }

    fn visit_pat(&mut self, p: &mut P<Pat>) {
        noop_visit_pat(p, self);
    }

    fn visit_anon_const(&mut self, c: &mut AnonConst) {
        noop_visit_anon_const(c, self);
    }

    fn visit_expr(&mut self, e: &mut P<Expr>) {
        noop_visit_expr(e, self);
    }

    /// This method is a hack to workaround unstable of `stmt_expr_attributes`.
    /// It can be removed once that feature is stabilized.
    fn visit_method_receiver_expr(&mut self, ex: &mut P<Expr>) {
        self.visit_expr(ex)
    }

    fn filter_map_expr(&mut self, e: P<Expr>) -> Option<P<Expr>> {
        noop_filter_map_expr(e, self)
    }

    fn visit_generic_arg(&mut self, arg: &mut GenericArg) {
        noop_visit_generic_arg(arg, self);
    }

    fn visit_ty(&mut self, t: &mut P<Ty>) {
        noop_visit_ty(t, self);
    }

    fn visit_lifetime(&mut self, l: &mut Lifetime) {
        noop_visit_lifetime(l, self);
    }

    fn visit_constraint(&mut self, t: &mut AssocConstraint) {
        noop_visit_constraint(t, self);
    }

    fn visit_foreign_mod(&mut self, nm: &mut ForeignMod) {
        noop_visit_foreign_mod(nm, self);
    }

    fn flat_map_variant(&mut self, v: Variant) -> SmallVec<[Variant; 1]> {
        noop_flat_map_variant(v, self)
    }

    fn visit_ident(&mut self, i: &mut Ident) {
        noop_visit_ident(i, self);
    }

    fn visit_path(&mut self, p: &mut Path) {
        noop_visit_path(p, self);
    }

    fn visit_qself(&mut self, qs: &mut Option<P<QSelf>>) {
        noop_visit_qself(qs, self);
    }

    fn visit_generic_args(&mut self, p: &mut GenericArgs) {
        noop_visit_generic_args(p, self);
    }

    fn visit_angle_bracketed_parameter_data(&mut self, p: &mut AngleBracketedArgs) {
        noop_visit_angle_bracketed_parameter_data(p, self);
    }

    fn visit_parenthesized_parameter_data(&mut self, p: &mut ParenthesizedArgs) {
        noop_visit_parenthesized_parameter_data(p, self);
    }

    fn visit_local(&mut self, l: &mut P<Local>) {
        noop_visit_local(l, self);
    }

    fn visit_mac_call(&mut self, mac: &mut MacCall) {
        noop_visit_mac(mac, self);
    }

    fn visit_macro_def(&mut self, def: &mut MacroDef) {
        noop_visit_macro_def(def, self);
    }

    fn visit_label(&mut self, label: &mut Label) {
        noop_visit_label(label, self);
    }

    fn visit_attribute(&mut self, at: &mut Attribute) {
        noop_visit_attribute(at, self);
    }

    fn flat_map_param(&mut self, param: Param) -> SmallVec<[Param; 1]> {
        noop_flat_map_param(param, self)
    }

    fn visit_generics(&mut self, generics: &mut Generics) {
        noop_visit_generics(generics, self);
    }

    fn visit_trait_ref(&mut self, tr: &mut TraitRef) {
        noop_visit_trait_ref(tr, self);
    }

    fn visit_poly_trait_ref(&mut self, p: &mut PolyTraitRef) {
        noop_visit_poly_trait_ref(p, self);
    }

    fn visit_variant_data(&mut self, vdata: &mut VariantData) {
        noop_visit_variant_data(vdata, self);
    }

    fn flat_map_generic_param(&mut self, param: GenericParam) -> SmallVec<[GenericParam; 1]> {
        noop_flat_map_generic_param(param, self)
    }

    fn visit_param_bound(&mut self, tpb: &mut GenericBound) {
        noop_visit_param_bound(tpb, self);
    }

    fn visit_mt(&mut self, mt: &mut MutTy) {
        noop_visit_mt(mt, self);
    }

    fn flat_map_expr_field(&mut self, f: ExprField) -> SmallVec<[ExprField; 1]> {
        noop_flat_map_expr_field(f, self)
    }

    fn visit_where_clause(&mut self, where_clause: &mut WhereClause) {
        noop_visit_where_clause(where_clause, self);
    }

    fn visit_where_predicate(&mut self, where_predicate: &mut WherePredicate) {
        noop_visit_where_predicate(where_predicate, self);
    }

    fn visit_vis(&mut self, vis: &mut Visibility) {
        noop_visit_vis(vis, self);
    }

    fn visit_id(&mut self, _id: &mut NodeId) {
        // Do nothing.
    }

    fn visit_span(&mut self, _sp: &mut Span) {
        // Do nothing.
    }

    fn flat_map_pat_field(&mut self, fp: PatField) -> SmallVec<[PatField; 1]> {
        noop_flat_map_pat_field(fp, self)
    }

    fn visit_inline_asm(&mut self, asm: &mut InlineAsm) {
        noop_visit_inline_asm(asm, self)
    }

    fn visit_inline_asm_sym(&mut self, sym: &mut InlineAsmSym) {
        noop_visit_inline_asm_sym(sym, self)
    }

    fn visit_format_args(&mut self, fmt: &mut FormatArgs) {
        noop_visit_format_args(fmt, self)
    }
}

/// Use a map-style function (`FnOnce(T) -> T`) to overwrite a `&mut T`. Useful
/// when using a `flat_map_*` or `filter_map_*` method within a `visit_`
/// method.
//
// No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
pub fn visit_clobber<T: DummyAstNode>(t: &mut T, f: impl FnOnce(T) -> T) {
    unsafe {
        // Safe because `t` is used in a read-only fashion by `read()` before
        // being overwritten by `write()`.
        let old_t = ptr::read(t);
        let new_t =
            panic::catch_unwind(panic::AssertUnwindSafe(|| f(old_t))).unwrap_or_else(|err| {
                // Set `t` to some valid but possible meaningless value,
                // and pass the fatal error further.
                ptr::write(t, T::dummy());
                panic::resume_unwind(err);
            });
        ptr::write(t, new_t);
    }
}

// No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
#[inline]
pub fn visit_vec<T, F>(elems: &mut Vec<T>, mut visit_elem: F)
where
    F: FnMut(&mut T),
{
    for elem in elems {
        visit_elem(elem);
    }
}

// No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
#[inline]
pub fn visit_thin_vec<T, F>(elems: &mut ThinVec<T>, mut visit_elem: F)
where
    F: FnMut(&mut T),
{
    for elem in elems {
        visit_elem(elem);
    }
}

// No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
#[inline]
pub fn visit_opt<T, F>(opt: &mut Option<T>, mut visit_elem: F)
where
    F: FnMut(&mut T),
{
    if let Some(elem) = opt {
        visit_elem(elem);
    }
}

// No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
pub fn visit_attrs<T: MutVisitor>(attrs: &mut AttrVec, vis: &mut T) {
    for attr in attrs.iter_mut() {
        vis.visit_attribute(attr);
    }
}

// No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
pub fn visit_exprs<T: MutVisitor>(exprs: &mut Vec<P<Expr>>, vis: &mut T) {
    exprs.flat_map_in_place(|expr| vis.filter_map_expr(expr))
}

// No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
pub fn visit_thin_exprs<T: MutVisitor>(exprs: &mut ThinVec<P<Expr>>, vis: &mut T) {
    exprs.flat_map_in_place(|expr| vis.filter_map_expr(expr))
}

// No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
pub fn visit_bounds<T: MutVisitor>(bounds: &mut GenericBounds, vis: &mut T) {
    visit_vec(bounds, |bound| vis.visit_param_bound(bound));
}

// No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
pub fn visit_fn_sig<T: MutVisitor>(FnSig { header, decl, span }: &mut FnSig, vis: &mut T) {
    vis.visit_fn_header(header);
    vis.visit_fn_decl(decl);
    vis.visit_span(span);
}

// No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
pub fn visit_attr_args<T: MutVisitor>(args: &mut AttrArgs, vis: &mut T) {
    match args {
        AttrArgs::Empty => {}
        AttrArgs::Delimited(args) => visit_delim_args(args, vis),
        AttrArgs::Eq(eq_span, AttrArgsEq::Ast(expr)) => {
            vis.visit_span(eq_span);
            vis.visit_expr(expr);
        }
        AttrArgs::Eq(_, AttrArgsEq::Hir(lit)) => {
            unreachable!("in literal form when visiting mac args eq: {:?}", lit)
        }
    }
}

// No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
pub fn visit_delim_args<T: MutVisitor>(args: &mut DelimArgs, vis: &mut T) {
    let DelimArgs { dspan, delim: _, tokens } = args;
    visit_delim_span(dspan, vis);
    visit_tts(tokens, vis);
}

pub fn visit_delim_span<T: MutVisitor>(dspan: &mut DelimSpan, vis: &mut T) {
    vis.visit_span(&mut dspan.open);
    vis.visit_span(&mut dspan.close);
}

pub fn noop_flat_map_pat_field<T: MutVisitor>(
    mut fp: PatField,
    vis: &mut T,
) -> SmallVec<[PatField; 1]> {
    let PatField { attrs, id, ident, is_placeholder: _, is_shorthand: _, pat, span } = &mut fp;
    vis.visit_id(id);
    vis.visit_ident(ident);
    vis.visit_pat(pat);
    vis.visit_span(span);
    visit_attrs(attrs, vis);
    smallvec![fp]
}

pub fn noop_visit_use_tree<T: MutVisitor>(use_tree: &mut UseTree, vis: &mut T) {
    let UseTree { prefix, kind, span } = use_tree;
    vis.visit_path(prefix);
    match kind {
        UseTreeKind::Simple(rename) => visit_opt(rename, |rename| vis.visit_ident(rename)),
        UseTreeKind::Nested(items) => {
            for (tree, id) in items {
                vis.visit_use_tree(tree);
                vis.visit_id(id);
            }
        }
        UseTreeKind::Glob => {}
    }
    vis.visit_span(span);
}

pub fn noop_flat_map_arm<T: MutVisitor>(mut arm: Arm, vis: &mut T) -> SmallVec<[Arm; 1]> {
    let Arm { attrs, pat, guard, body, span, id, is_placeholder: _ } = &mut arm;
    visit_attrs(attrs, vis);
    vis.visit_id(id);
    vis.visit_pat(pat);
    visit_opt(guard, |guard| vis.visit_expr(guard));
    vis.visit_expr(body);
    vis.visit_span(span);
    smallvec![arm]
}

pub fn noop_visit_constraint<T: MutVisitor>(
    AssocConstraint { id, ident, gen_args, kind, span }: &mut AssocConstraint,
    vis: &mut T,
) {
    vis.visit_id(id);
    vis.visit_ident(ident);
    if let Some(gen_args) = gen_args {
        vis.visit_generic_args(gen_args);
    }
    match kind {
        AssocConstraintKind::Equality { term } => match term {
            Term::Ty(ty) => vis.visit_ty(ty),
            Term::Const(c) => vis.visit_anon_const(c),
        },
        AssocConstraintKind::Bound { bounds } => visit_bounds(bounds, vis),
    }
    vis.visit_span(span);
}

pub fn noop_visit_ty<T: MutVisitor>(ty: &mut P<Ty>, vis: &mut T) {
    let Ty { id, kind, span, tokens } = ty.deref_mut();
    vis.visit_id(id);
    match kind {
        TyKind::Infer | TyKind::ImplicitSelf | TyKind::Err | TyKind::Never | TyKind::CVarArgs => {}
        TyKind::Slice(ty) => vis.visit_ty(ty),
        TyKind::Ptr(mt) => vis.visit_mt(mt),
        TyKind::Ref(lt, mt) => {
            visit_opt(lt, |lt| noop_visit_lifetime(lt, vis));
            vis.visit_mt(mt);
        }
        TyKind::BareFn(bft) => {
            let BareFnTy { unsafety, ext: _, generic_params, decl, decl_span } = bft.deref_mut();
            visit_unsafety(unsafety, vis);
            generic_params.flat_map_in_place(|param| vis.flat_map_generic_param(param));
            vis.visit_fn_decl(decl);
            vis.visit_span(decl_span);
        }
        TyKind::Tup(tys) => visit_thin_vec(tys, |ty| vis.visit_ty(ty)),
        TyKind::Paren(ty) => vis.visit_ty(ty),
        TyKind::Path(qself, path) => {
            vis.visit_qself(qself);
            vis.visit_path(path);
        }
        TyKind::Array(ty, length) => {
            vis.visit_ty(ty);
            vis.visit_anon_const(length);
        }
        TyKind::Typeof(expr) => vis.visit_anon_const(expr),
        TyKind::TraitObject(bounds, _syntax) => {
            visit_vec(bounds, |bound| vis.visit_param_bound(bound))
        }
        TyKind::ImplTrait(id, bounds) => {
            vis.visit_id(id);
            visit_vec(bounds, |bound| vis.visit_param_bound(bound));
        }
        TyKind::MacCall(mac) => vis.visit_mac_call(mac),
    }
    vis.visit_span(span);
    visit_lazy_tts(tokens, vis);
}

pub fn noop_visit_foreign_mod<T: MutVisitor>(foreign_mod: &mut ForeignMod, vis: &mut T) {
    let ForeignMod { unsafety, abi: _, items } = foreign_mod;
    visit_unsafety(unsafety, vis);
    items.flat_map_in_place(|item| vis.flat_map_foreign_item(item));
}

pub fn noop_flat_map_variant<T: MutVisitor>(
    mut variant: Variant,
    visitor: &mut T,
) -> SmallVec<[Variant; 1]> {
    let Variant { ident, vis, attrs, id, data, disr_expr, span, is_placeholder: _ } = &mut variant;
    visitor.visit_ident(ident);
    visitor.visit_vis(vis);
    visit_attrs(attrs, visitor);
    visitor.visit_id(id);
    visitor.visit_variant_data(data);
    visit_opt(disr_expr, |disr_expr| visitor.visit_anon_const(disr_expr));
    visitor.visit_span(span);
    smallvec![variant]
}

pub fn noop_visit_ident<T: MutVisitor>(Ident { name: _, span }: &mut Ident, vis: &mut T) {
    vis.visit_span(span);
}

pub fn noop_visit_path<T: MutVisitor>(Path { segments, span, tokens }: &mut Path, vis: &mut T) {
    vis.visit_span(span);
    for PathSegment { ident, id, args } in segments {
        vis.visit_ident(ident);
        vis.visit_id(id);
        visit_opt(args, |args| vis.visit_generic_args(args));
    }
    visit_lazy_tts(tokens, vis);
}

pub fn noop_visit_qself<T: MutVisitor>(qself: &mut Option<P<QSelf>>, vis: &mut T) {
    visit_opt(qself, |qself| {
        let QSelf { ty, path_span, position: _ } = &mut **qself;
        vis.visit_ty(ty);
        vis.visit_span(path_span);
    })
}

pub fn noop_visit_generic_args<T: MutVisitor>(generic_args: &mut GenericArgs, vis: &mut T) {
    match generic_args {
        GenericArgs::AngleBracketed(data) => vis.visit_angle_bracketed_parameter_data(data),
        GenericArgs::Parenthesized(data) => vis.visit_parenthesized_parameter_data(data),
    }
}

pub fn noop_visit_generic_arg<T: MutVisitor>(arg: &mut GenericArg, vis: &mut T) {
    match arg {
        GenericArg::Lifetime(lt) => vis.visit_lifetime(lt),
        GenericArg::Type(ty) => vis.visit_ty(ty),
        GenericArg::Const(ct) => vis.visit_anon_const(ct),
    }
}

pub fn noop_visit_angle_bracketed_parameter_data<T: MutVisitor>(
    data: &mut AngleBracketedArgs,
    vis: &mut T,
) {
    let AngleBracketedArgs { args, span } = data;
    visit_thin_vec(args, |arg| match arg {
        AngleBracketedArg::Arg(arg) => vis.visit_generic_arg(arg),
        AngleBracketedArg::Constraint(constraint) => vis.visit_constraint(constraint),
    });
    vis.visit_span(span);
}

pub fn noop_visit_parenthesized_parameter_data<T: MutVisitor>(
    args: &mut ParenthesizedArgs,
    vis: &mut T,
) {
    let ParenthesizedArgs { inputs, output, span, .. } = args;
    visit_thin_vec(inputs, |input| vis.visit_ty(input));
    noop_visit_fn_ret_ty(output, vis);
    vis.visit_span(span);
}

pub fn noop_visit_local<T: MutVisitor>(local: &mut P<Local>, vis: &mut T) {
    let Local { id, pat, ty, kind, span, attrs, tokens } = local.deref_mut();
    vis.visit_id(id);
    vis.visit_pat(pat);
    visit_opt(ty, |ty| vis.visit_ty(ty));
    match kind {
        LocalKind::Decl => {}
        LocalKind::Init(init) => {
            vis.visit_expr(init);
        }
        LocalKind::InitElse(init, els) => {
            vis.visit_expr(init);
            vis.visit_block(els);
        }
    }
    vis.visit_span(span);
    visit_attrs(attrs, vis);
    visit_lazy_tts(tokens, vis);
}

pub fn noop_visit_attribute<T: MutVisitor>(attr: &mut Attribute, vis: &mut T) {
    let Attribute { kind, id: _, style: _, span } = attr;
    match kind {
        AttrKind::Normal(normal) => {
            let NormalAttr { item: AttrItem { path, args, tokens }, tokens: attr_tokens } =
                &mut **normal;
            vis.visit_path(path);
            visit_attr_args(args, vis);
            visit_lazy_tts(tokens, vis);
            visit_lazy_tts(attr_tokens, vis);
        }
        AttrKind::DocComment(..) => {}
    }
    vis.visit_span(span);
}

pub fn noop_visit_mac<T: MutVisitor>(mac: &mut MacCall, vis: &mut T) {
    let MacCall { path, args, prior_type_ascription: _ } = mac;
    vis.visit_path(path);
    visit_delim_args(args, vis);
}

pub fn noop_visit_macro_def<T: MutVisitor>(macro_def: &mut MacroDef, vis: &mut T) {
    let MacroDef { body, macro_rules: _ } = macro_def;
    visit_delim_args(body, vis);
}

pub fn noop_visit_meta_list_item<T: MutVisitor>(li: &mut NestedMetaItem, vis: &mut T) {
    match li {
        NestedMetaItem::MetaItem(mi) => vis.visit_meta_item(mi),
        NestedMetaItem::Lit(_lit) => {}
    }
}

pub fn noop_visit_meta_item<T: MutVisitor>(mi: &mut MetaItem, vis: &mut T) {
    let MetaItem { path: _, kind, span } = mi;
    match kind {
        MetaItemKind::Word => {}
        MetaItemKind::List(mis) => visit_thin_vec(mis, |mi| vis.visit_meta_list_item(mi)),
        MetaItemKind::NameValue(_s) => {}
    }
    vis.visit_span(span);
}

pub fn noop_flat_map_param<T: MutVisitor>(mut param: Param, vis: &mut T) -> SmallVec<[Param; 1]> {
    let Param { attrs, id, pat, span, ty, is_placeholder: _ } = &mut param;
    vis.visit_id(id);
    visit_attrs(attrs, vis);
    vis.visit_pat(pat);
    vis.visit_span(span);
    vis.visit_ty(ty);
    smallvec![param]
}

// No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
pub fn visit_attr_tt<T: MutVisitor>(tt: &mut AttrTokenTree, vis: &mut T) {
    match tt {
        AttrTokenTree::Token(token, _) => {
            visit_token(token, vis);
        }
        AttrTokenTree::Delimited(DelimSpan { open, close }, _delim, tts) => {
            vis.visit_span(open);
            vis.visit_span(close);
            visit_attr_tts(tts, vis);
        }
        AttrTokenTree::Attributes(data) => {
            for attr in &mut *data.attrs {
                match &mut attr.kind {
                    AttrKind::Normal(normal) => {
                        visit_lazy_tts(&mut normal.tokens, vis);
                    }
                    AttrKind::DocComment(..) => {
                        vis.visit_span(&mut attr.span);
                    }
                }
            }
            visit_lazy_tts_opt_mut(Some(&mut data.tokens), vis);
        }
    }
}

// No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
pub fn visit_tt<T: MutVisitor>(tt: &mut TokenTree, vis: &mut T) {
    match tt {
        TokenTree::Token(token, _) => {
            visit_token(token, vis);
        }
        TokenTree::Delimited(DelimSpan { open, close }, _delim, tts) => {
            vis.visit_span(open);
            vis.visit_span(close);
            visit_tts(tts, vis);
        }
    }
}

// No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
pub fn visit_tts<T: MutVisitor>(TokenStream(tts): &mut TokenStream, vis: &mut T) {
    if T::VISIT_TOKENS && !tts.is_empty() {
        let tts = Lrc::make_mut(tts);
        visit_vec(tts, |tree| visit_tt(tree, vis));
    }
}

pub fn visit_attr_tts<T: MutVisitor>(AttrTokenStream(tts): &mut AttrTokenStream, vis: &mut T) {
    if T::VISIT_TOKENS && !tts.is_empty() {
        let tts = Lrc::make_mut(tts);
        visit_vec(tts, |tree| visit_attr_tt(tree, vis));
    }
}

pub fn visit_lazy_tts_opt_mut<T: MutVisitor>(
    lazy_tts: Option<&mut LazyAttrTokenStream>,
    vis: &mut T,
) {
    if T::VISIT_TOKENS {
        if let Some(lazy_tts) = lazy_tts {
            let mut tts = lazy_tts.to_attr_token_stream();
            visit_attr_tts(&mut tts, vis);
            *lazy_tts = LazyAttrTokenStream::new(tts);
        }
    }
}

pub fn visit_lazy_tts<T: MutVisitor>(lazy_tts: &mut Option<LazyAttrTokenStream>, vis: &mut T) {
    visit_lazy_tts_opt_mut(lazy_tts.as_mut(), vis);
}

/// Applies ident visitor if it's an ident; applies other visits to interpolated nodes.
/// In practice the ident part is not actually used by specific visitors right now,
/// but there's a test below checking that it works.
// No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
pub fn visit_token<T: MutVisitor>(t: &mut Token, vis: &mut T) {
    let Token { kind, span } = t;
    match kind {
        token::Ident(name, _) | token::Lifetime(name) => {
            let mut ident = Ident::new(*name, *span);
            vis.visit_ident(&mut ident);
            *name = ident.name;
            *span = ident.span;
            return; // Avoid visiting the span for the second time.
        }
        token::Interpolated(nt) => {
            visit_nonterminal(Lrc::make_mut(nt), vis);
        }
        _ => {}
    }
    vis.visit_span(span);
}

// No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
/// Applies the visitor to elements of interpolated nodes.
//
// N.B., this can occur only when applying a visitor to partially expanded
// code, where parsed pieces have gotten implanted ito *other* macro
// invocations. This is relevant for macro hygiene, but possibly not elsewhere.
//
// One problem here occurs because the types for flat_map_item, flat_map_stmt,
// etc., allow the visitor to return *multiple* items; this is a problem for the
// nodes here, because they insist on having exactly one piece. One solution
// would be to mangle the MutVisitor trait to include one-to-many and
// one-to-one versions of these entry points, but that would probably confuse a
// lot of people and help very few. Instead, I'm just going to put in dynamic
// checks. I think the performance impact of this will be pretty much
// nonexistent. The danger is that someone will apply a `MutVisitor` to a
// partially expanded node, and will be confused by the fact that their
// `flat_map_item` or `flat_map_stmt` isn't getting called on `NtItem` or `NtStmt`
// nodes. Hopefully they'll wind up reading this comment, and doing something
// appropriate.
//
// BTW, design choice: I considered just changing the type of, e.g., `NtItem` to
// contain multiple items, but decided against it when I looked at
// `parse_item_or_view_item` and tried to figure out what I would do with
// multiple items there....
pub fn visit_nonterminal<T: MutVisitor>(nt: &mut token::Nonterminal, vis: &mut T) {
    match nt {
        token::NtItem(item) => visit_clobber(item, |item| {
            // This is probably okay, because the only visitors likely to
            // peek inside interpolated nodes will be renamings/markings,
            // which map single items to single items.
            vis.flat_map_item(item).expect_one("expected visitor to produce exactly one item")
        }),
        token::NtBlock(block) => vis.visit_block(block),
        token::NtStmt(stmt) => visit_clobber(stmt, |stmt| {
            // See reasoning above.
            stmt.map(|stmt| {
                vis.flat_map_stmt(stmt).expect_one("expected visitor to produce exactly one item")
            })
        }),
        token::NtPat(pat) => vis.visit_pat(pat),
        token::NtExpr(expr) => vis.visit_expr(expr),
        token::NtTy(ty) => vis.visit_ty(ty),
        token::NtIdent(ident, _is_raw) => vis.visit_ident(ident),
        token::NtLifetime(ident) => vis.visit_ident(ident),
        token::NtLiteral(expr) => vis.visit_expr(expr),
        token::NtMeta(item) => {
            let AttrItem { path, args, tokens } = item.deref_mut();
            vis.visit_path(path);
            visit_attr_args(args, vis);
            visit_lazy_tts(tokens, vis);
        }
        token::NtPath(path) => vis.visit_path(path),
        token::NtVis(visib) => vis.visit_vis(visib),
    }
}

// No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
pub fn visit_defaultness<T: MutVisitor>(defaultness: &mut Defaultness, vis: &mut T) {
    match defaultness {
        Defaultness::Default(span) => vis.visit_span(span),
        Defaultness::Final => {}
    }
}

// No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
pub fn visit_unsafety<T: MutVisitor>(unsafety: &mut Unsafe, vis: &mut T) {
    match unsafety {
        Unsafe::Yes(span) => vis.visit_span(span),
        Unsafe::No => {}
    }
}

// No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
pub fn visit_polarity<T: MutVisitor>(polarity: &mut ImplPolarity, vis: &mut T) {
    match polarity {
        ImplPolarity::Positive => {}
        ImplPolarity::Negative(span) => vis.visit_span(span),
    }
}

// No `noop_` prefix because there isn't a corresponding method in `MutVisitor`.
pub fn visit_constness<T: MutVisitor>(constness: &mut Const, vis: &mut T) {
    match constness {
        Const::Yes(span) => vis.visit_span(span),
        Const::No => {}
    }
}

pub fn noop_visit_closure_binder<T: MutVisitor>(binder: &mut ClosureBinder, vis: &mut T) {
    match binder {
        ClosureBinder::NotPresent => {}
        ClosureBinder::For { span: _, generic_params } => {
            generic_params.flat_map_in_place(|param| vis.flat_map_generic_param(param));
        }
    }
}

pub fn noop_visit_asyncness<T: MutVisitor>(asyncness: &mut Async, vis: &mut T) {
    match asyncness {
        Async::Yes { span: _, closure_id, return_impl_trait_id } => {
            vis.visit_id(closure_id);
            vis.visit_id(return_impl_trait_id);
        }
        Async::No => {}
    }
}

pub fn noop_visit_fn_decl<T: MutVisitor>(decl: &mut P<FnDecl>, vis: &mut T) {
    let FnDecl { inputs, output } = decl.deref_mut();
    inputs.flat_map_in_place(|param| vis.flat_map_param(param));
    noop_visit_fn_ret_ty(output, vis);
}

pub fn noop_visit_fn_ret_ty<T: MutVisitor>(fn_ret_ty: &mut FnRetTy, vis: &mut T) {
    match fn_ret_ty {
        FnRetTy::Default(span) => vis.visit_span(span),
        FnRetTy::Ty(ty) => vis.visit_ty(ty),
    }
}

pub fn noop_visit_param_bound<T: MutVisitor>(pb: &mut GenericBound, vis: &mut T) {
    match pb {
        GenericBound::Trait(ty, _modifier) => vis.visit_poly_trait_ref(ty),
        GenericBound::Outlives(lifetime) => noop_visit_lifetime(lifetime, vis),
    }
}

pub fn noop_flat_map_generic_param<T: MutVisitor>(
    mut param: GenericParam,
    vis: &mut T,
) -> SmallVec<[GenericParam; 1]> {
    let GenericParam { id, ident, attrs, bounds, kind, colon_span, is_placeholder: _ } = &mut param;
    vis.visit_id(id);
    vis.visit_ident(ident);
    if let Some(colon_span) = colon_span {
        vis.visit_span(colon_span);
    }
    visit_attrs(attrs, vis);
    visit_vec(bounds, |bound| noop_visit_param_bound(bound, vis));
    match kind {
        GenericParamKind::Lifetime => {}
        GenericParamKind::Type { default } => {
            visit_opt(default, |default| vis.visit_ty(default));
        }
        GenericParamKind::Const { ty, kw_span: _, default } => {
            vis.visit_ty(ty);
            visit_opt(default, |default| vis.visit_anon_const(default));
        }
    }
    smallvec![param]
}

pub fn noop_visit_label<T: MutVisitor>(Label { ident }: &mut Label, vis: &mut T) {
    vis.visit_ident(ident);
}

fn noop_visit_lifetime<T: MutVisitor>(Lifetime { id, ident }: &mut Lifetime, vis: &mut T) {
    vis.visit_id(id);
    vis.visit_ident(ident);
}

pub fn noop_visit_generics<T: MutVisitor>(generics: &mut Generics, vis: &mut T) {
    let Generics { params, where_clause, span } = generics;
    params.flat_map_in_place(|param| vis.flat_map_generic_param(param));
    vis.visit_where_clause(where_clause);
    vis.visit_span(span);
}

pub fn noop_visit_where_clause<T: MutVisitor>(wc: &mut WhereClause, vis: &mut T) {
    let WhereClause { has_where_token: _, predicates, span } = wc;
    visit_thin_vec(predicates, |predicate| vis.visit_where_predicate(predicate));
    vis.visit_span(span);
}

pub fn noop_visit_where_predicate<T: MutVisitor>(pred: &mut WherePredicate, vis: &mut T) {
    match pred {
        WherePredicate::BoundPredicate(bp) => {
            let WhereBoundPredicate { span, bound_generic_params, bounded_ty, bounds } = bp;
            vis.visit_span(span);
            bound_generic_params.flat_map_in_place(|param| vis.flat_map_generic_param(param));
            vis.visit_ty(bounded_ty);
            visit_vec(bounds, |bound| vis.visit_param_bound(bound));
        }
        WherePredicate::RegionPredicate(rp) => {
            let WhereRegionPredicate { span, lifetime, bounds } = rp;
            vis.visit_span(span);
            noop_visit_lifetime(lifetime, vis);
            visit_vec(bounds, |bound| noop_visit_param_bound(bound, vis));
        }
        WherePredicate::EqPredicate(ep) => {
            let WhereEqPredicate { span, lhs_ty, rhs_ty } = ep;
            vis.visit_span(span);
            vis.visit_ty(lhs_ty);
            vis.visit_ty(rhs_ty);
        }
    }
}

pub fn noop_visit_variant_data<T: MutVisitor>(vdata: &mut VariantData, vis: &mut T) {
    match vdata {
        VariantData::Struct(fields, ..) => {
            fields.flat_map_in_place(|field| vis.flat_map_field_def(field));
        }
        VariantData::Tuple(fields, id) => {
            fields.flat_map_in_place(|field| vis.flat_map_field_def(field));
            vis.visit_id(id);
        }
        VariantData::Unit(id) => vis.visit_id(id),
    }
}

pub fn noop_visit_trait_ref<T: MutVisitor>(TraitRef { path, ref_id }: &mut TraitRef, vis: &mut T) {
    vis.visit_path(path);
    vis.visit_id(ref_id);
}

pub fn noop_visit_poly_trait_ref<T: MutVisitor>(p: &mut PolyTraitRef, vis: &mut T) {
    let PolyTraitRef { bound_generic_params, trait_ref, span } = p;
    bound_generic_params.flat_map_in_place(|param| vis.flat_map_generic_param(param));
    vis.visit_trait_ref(trait_ref);
    vis.visit_span(span);
}

pub fn noop_flat_map_field_def<T: MutVisitor>(
    mut fd: FieldDef,
    visitor: &mut T,
) -> SmallVec<[FieldDef; 1]> {
    let FieldDef { span, ident, vis, id, ty, attrs, is_placeholder: _ } = &mut fd;
    visitor.visit_span(span);
    visit_opt(ident, |ident| visitor.visit_ident(ident));
    visitor.visit_vis(vis);
    visitor.visit_id(id);
    visitor.visit_ty(ty);
    visit_attrs(attrs, visitor);
    smallvec![fd]
}

pub fn noop_flat_map_expr_field<T: MutVisitor>(
    mut f: ExprField,
    vis: &mut T,
) -> SmallVec<[ExprField; 1]> {
    let ExprField { ident, expr, span, is_shorthand: _, attrs, id, is_placeholder: _ } = &mut f;
    vis.visit_ident(ident);
    vis.visit_expr(expr);
    vis.visit_id(id);
    vis.visit_span(span);
    visit_attrs(attrs, vis);
    smallvec![f]
}

pub fn noop_visit_mt<T: MutVisitor>(MutTy { ty, mutbl: _ }: &mut MutTy, vis: &mut T) {
    vis.visit_ty(ty);
}

pub fn noop_visit_block<T: MutVisitor>(block: &mut P<Block>, vis: &mut T) {
    let Block { id, stmts, rules: _, span, tokens, could_be_bare_literal: _ } = block.deref_mut();
    vis.visit_id(id);
    stmts.flat_map_in_place(|stmt| vis.flat_map_stmt(stmt));
    vis.visit_span(span);
    visit_lazy_tts(tokens, vis);
}

pub fn noop_visit_item_kind<T: MutVisitor>(kind: &mut ItemKind, vis: &mut T) {
    match kind {
        ItemKind::ExternCrate(_orig_name) => {}
        ItemKind::Use(use_tree) => vis.visit_use_tree(use_tree),
        ItemKind::Static(ty, _, expr) => {
            vis.visit_ty(ty);
            visit_opt(expr, |expr| vis.visit_expr(expr));
        }
        ItemKind::Const(defaultness, ty, expr) => {
            visit_defaultness(defaultness, vis);
            vis.visit_ty(ty);
            visit_opt(expr, |expr| vis.visit_expr(expr));
        }
        ItemKind::Fn(box Fn { defaultness, generics, sig, body }) => {
            visit_defaultness(defaultness, vis);
            visit_fn_sig(sig, vis);
            vis.visit_generics(generics);
            visit_opt(body, |body| vis.visit_block(body));
        }
        ItemKind::Mod(unsafety, mod_kind) => {
            visit_unsafety(unsafety, vis);
            match mod_kind {
                ModKind::Loaded(items, _inline, ModSpans { inner_span, inject_use_span }) => {
                    vis.visit_span(inner_span);
                    vis.visit_span(inject_use_span);
                    items.flat_map_in_place(|item| vis.flat_map_item(item));
                }
                ModKind::Unloaded => {}
            }
        }
        ItemKind::ForeignMod(nm) => vis.visit_foreign_mod(nm),
        ItemKind::GlobalAsm(asm) => vis.visit_inline_asm(asm),
        ItemKind::TyAlias(box TyAlias {
            defaultness, generics, where_clauses, bounds, ty, ..
        }) => {
            visit_defaultness(defaultness, vis);
            vis.visit_generics(generics);
            vis.visit_span(&mut where_clauses.0.1);
            vis.visit_span(&mut where_clauses.1.1);
            visit_bounds(bounds, vis);
            visit_opt(ty, |ty| vis.visit_ty(ty));
        }
        ItemKind::Enum(EnumDef { variants }, generics) => {
            variants.flat_map_in_place(|variant| vis.flat_map_variant(variant));
            vis.visit_generics(generics);
        }
        ItemKind::Struct(variant_data, generics) | ItemKind::Union(variant_data, generics) => {
            vis.visit_variant_data(variant_data);
            vis.visit_generics(generics);
        }
        ItemKind::Impl(box Impl {
            defaultness,
            unsafety,
            generics,
            constness,
            polarity,
            of_trait,
            self_ty,
            items,
        }) => {
            visit_defaultness(defaultness, vis);
            visit_unsafety(unsafety, vis);
            vis.visit_generics(generics);
            visit_constness(constness, vis);
            visit_polarity(polarity, vis);
            visit_opt(of_trait, |trait_ref| vis.visit_trait_ref(trait_ref));
            vis.visit_ty(self_ty);
            items.flat_map_in_place(|item| vis.flat_map_impl_item(item));
        }
        ItemKind::Trait(box Trait { unsafety, is_auto: _, generics, bounds, items }) => {
            visit_unsafety(unsafety, vis);
            vis.visit_generics(generics);
            visit_bounds(bounds, vis);
            items.flat_map_in_place(|item| vis.flat_map_trait_item(item));
        }
        ItemKind::TraitAlias(generics, bounds) => {
            vis.visit_generics(generics);
            visit_bounds(bounds, vis);
        }
        ItemKind::MacCall(m) => vis.visit_mac_call(m),
        ItemKind::MacroDef(def) => vis.visit_macro_def(def),
    }
}

pub fn noop_flat_map_assoc_item<T: MutVisitor>(
    mut item: P<AssocItem>,
    visitor: &mut T,
) -> SmallVec<[P<AssocItem>; 1]> {
    let Item { id, ident, vis, attrs, kind, span, tokens } = item.deref_mut();
    visitor.visit_id(id);
    visitor.visit_ident(ident);
    visitor.visit_vis(vis);
    visit_attrs(attrs, visitor);
    match kind {
        AssocItemKind::Const(defaultness, ty, expr) => {
            visit_defaultness(defaultness, visitor);
            visitor.visit_ty(ty);
            visit_opt(expr, |expr| visitor.visit_expr(expr));
        }
        AssocItemKind::Fn(box Fn { defaultness, generics, sig, body }) => {
            visit_defaultness(defaultness, visitor);
            visitor.visit_generics(generics);
            visit_fn_sig(sig, visitor);
            visit_opt(body, |body| visitor.visit_block(body));
        }
        AssocItemKind::Type(box TyAlias {
            defaultness,
            generics,
            where_clauses,
            bounds,
            ty,
            ..
        }) => {
            visit_defaultness(defaultness, visitor);
            visitor.visit_generics(generics);
            visitor.visit_span(&mut where_clauses.0.1);
            visitor.visit_span(&mut where_clauses.1.1);
            visit_bounds(bounds, visitor);
            visit_opt(ty, |ty| visitor.visit_ty(ty));
        }
        AssocItemKind::MacCall(mac) => visitor.visit_mac_call(mac),
    }
    visitor.visit_span(span);
    visit_lazy_tts(tokens, visitor);
    smallvec![item]
}

pub fn noop_visit_fn_header<T: MutVisitor>(header: &mut FnHeader, vis: &mut T) {
    let FnHeader { unsafety, asyncness, constness, ext: _ } = header;
    visit_constness(constness, vis);
    vis.visit_asyncness(asyncness);
    visit_unsafety(unsafety, vis);
}

pub fn noop_visit_crate<T: MutVisitor>(krate: &mut Crate, vis: &mut T) {
    let Crate { attrs, items, spans, id, is_placeholder: _ } = krate;
    vis.visit_id(id);
    visit_attrs(attrs, vis);
    items.flat_map_in_place(|item| vis.flat_map_item(item));
    let ModSpans { inner_span, inject_use_span } = spans;
    vis.visit_span(inner_span);
    vis.visit_span(inject_use_span);
}

// Mutates one item into possibly many items.
pub fn noop_flat_map_item<T: MutVisitor>(
    mut item: P<Item>,
    visitor: &mut T,
) -> SmallVec<[P<Item>; 1]> {
    let Item { ident, attrs, id, kind, vis, span, tokens } = item.deref_mut();
    visitor.visit_ident(ident);
    visit_attrs(attrs, visitor);
    visitor.visit_id(id);
    visitor.visit_item_kind(kind);
    visitor.visit_vis(vis);
    visitor.visit_span(span);
    visit_lazy_tts(tokens, visitor);

    smallvec![item]
}

pub fn noop_flat_map_foreign_item<T: MutVisitor>(
    mut item: P<ForeignItem>,
    visitor: &mut T,
) -> SmallVec<[P<ForeignItem>; 1]> {
    let Item { ident, attrs, id, kind, vis, span, tokens } = item.deref_mut();
    visitor.visit_id(id);
    visitor.visit_ident(ident);
    visitor.visit_vis(vis);
    visit_attrs(attrs, visitor);
    match kind {
        ForeignItemKind::Static(ty, _, expr) => {
            visitor.visit_ty(ty);
            visit_opt(expr, |expr| visitor.visit_expr(expr));
        }
        ForeignItemKind::Fn(box Fn { defaultness, generics, sig, body }) => {
            visit_defaultness(defaultness, visitor);
            visitor.visit_generics(generics);
            visit_fn_sig(sig, visitor);
            visit_opt(body, |body| visitor.visit_block(body));
        }
        ForeignItemKind::TyAlias(box TyAlias {
            defaultness,
            generics,
            where_clauses,
            bounds,
            ty,
            ..
        }) => {
            visit_defaultness(defaultness, visitor);
            visitor.visit_generics(generics);
            visitor.visit_span(&mut where_clauses.0.1);
            visitor.visit_span(&mut where_clauses.1.1);
            visit_bounds(bounds, visitor);
            visit_opt(ty, |ty| visitor.visit_ty(ty));
        }
        ForeignItemKind::MacCall(mac) => visitor.visit_mac_call(mac),
    }
    visitor.visit_span(span);
    visit_lazy_tts(tokens, visitor);
    smallvec![item]
}

pub fn noop_visit_pat<T: MutVisitor>(pat: &mut P<Pat>, vis: &mut T) {
    let Pat { id, kind, span, tokens } = pat.deref_mut();
    vis.visit_id(id);
    match kind {
        PatKind::Wild | PatKind::Rest => {}
        PatKind::Ident(_binding_mode, ident, sub) => {
            vis.visit_ident(ident);
            visit_opt(sub, |sub| vis.visit_pat(sub));
        }
        PatKind::Lit(e) => vis.visit_expr(e),
        PatKind::TupleStruct(qself, path, elems) => {
            vis.visit_qself(qself);
            vis.visit_path(path);
            visit_thin_vec(elems, |elem| vis.visit_pat(elem));
        }
        PatKind::Path(qself, path) => {
            vis.visit_qself(qself);
            vis.visit_path(path);
        }
        PatKind::Struct(qself, path, fields, _etc) => {
            vis.visit_qself(qself);
            vis.visit_path(path);
            fields.flat_map_in_place(|field| vis.flat_map_pat_field(field));
        }
        PatKind::Box(inner) => vis.visit_pat(inner),
        PatKind::Ref(inner, _mutbl) => vis.visit_pat(inner),
        PatKind::Range(e1, e2, Spanned { span: _, node: _ }) => {
            visit_opt(e1, |e| vis.visit_expr(e));
            visit_opt(e2, |e| vis.visit_expr(e));
            vis.visit_span(span);
        }
        PatKind::Tuple(elems) | PatKind::Slice(elems) | PatKind::Or(elems) => {
            visit_thin_vec(elems, |elem| vis.visit_pat(elem))
        }
        PatKind::Paren(inner) => vis.visit_pat(inner),
        PatKind::MacCall(mac) => vis.visit_mac_call(mac),
    }
    vis.visit_span(span);
    visit_lazy_tts(tokens, vis);
}

pub fn noop_visit_anon_const<T: MutVisitor>(AnonConst { id, value }: &mut AnonConst, vis: &mut T) {
    vis.visit_id(id);
    vis.visit_expr(value);
}

pub fn noop_visit_inline_asm<T: MutVisitor>(asm: &mut InlineAsm, vis: &mut T) {
    for (op, _) in &mut asm.operands {
        match op {
            InlineAsmOperand::In { expr, .. }
            | InlineAsmOperand::Out { expr: Some(expr), .. }
            | InlineAsmOperand::InOut { expr, .. } => vis.visit_expr(expr),
            InlineAsmOperand::Out { expr: None, .. } => {}
            InlineAsmOperand::SplitInOut { in_expr, out_expr, .. } => {
                vis.visit_expr(in_expr);
                if let Some(out_expr) = out_expr {
                    vis.visit_expr(out_expr);
                }
            }
            InlineAsmOperand::Const { anon_const } => vis.visit_anon_const(anon_const),
            InlineAsmOperand::Sym { sym } => vis.visit_inline_asm_sym(sym),
        }
    }
}

pub fn noop_visit_inline_asm_sym<T: MutVisitor>(
    InlineAsmSym { id, qself, path }: &mut InlineAsmSym,
    vis: &mut T,
) {
    vis.visit_id(id);
    vis.visit_qself(qself);
    vis.visit_path(path);
}

pub fn noop_visit_format_args<T: MutVisitor>(fmt: &mut FormatArgs, vis: &mut T) {
    for arg in fmt.arguments.all_args_mut() {
        if let FormatArgumentKind::Named(name) = &mut arg.kind {
            vis.visit_ident(name);
        }
        vis.visit_expr(&mut arg.expr);
    }
}

pub fn noop_visit_expr<T: MutVisitor>(
    Expr { kind, id, span, attrs, tokens }: &mut Expr,
    vis: &mut T,
) {
    match kind {
        ExprKind::Array(exprs) => visit_thin_exprs(exprs, vis),
        ExprKind::ConstBlock(anon_const) => {
            vis.visit_anon_const(anon_const);
        }
        ExprKind::Repeat(expr, count) => {
            vis.visit_expr(expr);
            vis.visit_anon_const(count);
        }
        ExprKind::Tup(exprs) => visit_thin_exprs(exprs, vis),
        ExprKind::Call(f, args) => {
            vis.visit_expr(f);
            visit_thin_exprs(args, vis);
        }
        ExprKind::MethodCall(box MethodCall {
            seg: PathSegment { ident, id, args: seg_args },
            receiver,
            args: call_args,
            span,
        }) => {
            vis.visit_ident(ident);
            vis.visit_id(id);
            visit_opt(seg_args, |args| vis.visit_generic_args(args));
            vis.visit_method_receiver_expr(receiver);
            visit_thin_exprs(call_args, vis);
            vis.visit_span(span);
        }
        ExprKind::Binary(_binop, lhs, rhs) => {
            vis.visit_expr(lhs);
            vis.visit_expr(rhs);
        }
        ExprKind::Unary(_unop, ohs) => vis.visit_expr(ohs),
        ExprKind::Cast(expr, ty) => {
            vis.visit_expr(expr);
            vis.visit_ty(ty);
        }
        ExprKind::Type(expr, ty) => {
            vis.visit_expr(expr);
            vis.visit_ty(ty);
        }
        ExprKind::AddrOf(_, _, ohs) => vis.visit_expr(ohs),
        ExprKind::Let(pat, scrutinee, _) => {
            vis.visit_pat(pat);
            vis.visit_expr(scrutinee);
        }
        ExprKind::If(cond, tr, fl) => {
            vis.visit_expr(cond);
            vis.visit_block(tr);
            visit_opt(fl, |fl| vis.visit_expr(fl));
        }
        ExprKind::While(cond, body, label) => {
            vis.visit_expr(cond);
            vis.visit_block(body);
            visit_opt(label, |label| vis.visit_label(label));
        }
        ExprKind::ForLoop(pat, iter, body, label) => {
            vis.visit_pat(pat);
            vis.visit_expr(iter);
            vis.visit_block(body);
            visit_opt(label, |label| vis.visit_label(label));
        }
        ExprKind::Loop(body, label, span) => {
            vis.visit_block(body);
            visit_opt(label, |label| vis.visit_label(label));
            vis.visit_span(span);
        }
        ExprKind::Match(expr, arms) => {
            vis.visit_expr(expr);
            arms.flat_map_in_place(|arm| vis.flat_map_arm(arm));
        }
        ExprKind::Closure(box Closure {
            binder,
            capture_clause: _,
            constness,
            asyncness,
            movability: _,
            fn_decl,
            body,
            fn_decl_span,
            fn_arg_span: _,
        }) => {
            vis.visit_closure_binder(binder);
            visit_constness(constness, vis);
            vis.visit_asyncness(asyncness);
            vis.visit_fn_decl(fn_decl);
            vis.visit_expr(body);
            vis.visit_span(fn_decl_span);
        }
        ExprKind::Block(blk, label) => {
            vis.visit_block(blk);
            visit_opt(label, |label| vis.visit_label(label));
        }
        ExprKind::Async(_capture_by, body) => {
            vis.visit_block(body);
        }
        ExprKind::Await(expr) => vis.visit_expr(expr),
        ExprKind::Assign(el, er, _) => {
            vis.visit_expr(el);
            vis.visit_expr(er);
        }
        ExprKind::AssignOp(_op, el, er) => {
            vis.visit_expr(el);
            vis.visit_expr(er);
        }
        ExprKind::Field(el, ident) => {
            vis.visit_expr(el);
            vis.visit_ident(ident);
        }
        ExprKind::Index(el, er) => {
            vis.visit_expr(el);
            vis.visit_expr(er);
        }
        ExprKind::Range(e1, e2, _lim) => {
            visit_opt(e1, |e1| vis.visit_expr(e1));
            visit_opt(e2, |e2| vis.visit_expr(e2));
        }
        ExprKind::Underscore => {}
        ExprKind::Path(qself, path) => {
            vis.visit_qself(qself);
            vis.visit_path(path);
        }
        ExprKind::Break(label, expr) => {
            visit_opt(label, |label| vis.visit_label(label));
            visit_opt(expr, |expr| vis.visit_expr(expr));
        }
        ExprKind::Continue(label) => {
            visit_opt(label, |label| vis.visit_label(label));
        }
        ExprKind::Ret(expr) => {
            visit_opt(expr, |expr| vis.visit_expr(expr));
        }
        ExprKind::Yeet(expr) => {
            visit_opt(expr, |expr| vis.visit_expr(expr));
        }
        ExprKind::InlineAsm(asm) => vis.visit_inline_asm(asm),
        ExprKind::FormatArgs(fmt) => vis.visit_format_args(fmt),
        ExprKind::MacCall(mac) => vis.visit_mac_call(mac),
        ExprKind::Struct(se) => {
            let StructExpr { qself, path, fields, rest } = se.deref_mut();
            vis.visit_qself(qself);
            vis.visit_path(path);
            fields.flat_map_in_place(|field| vis.flat_map_expr_field(field));
            match rest {
                StructRest::Base(expr) => vis.visit_expr(expr),
                StructRest::Rest(_span) => {}
                StructRest::None => {}
            }
        }
        ExprKind::Paren(expr) => {
            vis.visit_expr(expr);
        }
        ExprKind::Yield(expr) => {
            visit_opt(expr, |expr| vis.visit_expr(expr));
        }
        ExprKind::Try(expr) => vis.visit_expr(expr),
        ExprKind::TryBlock(body) => vis.visit_block(body),
        ExprKind::Lit(_) | ExprKind::IncludedBytes(..) | ExprKind::Err => {}
    }
    vis.visit_id(id);
    vis.visit_span(span);
    visit_attrs(attrs, vis);
    visit_lazy_tts(tokens, vis);
}

pub fn noop_filter_map_expr<T: MutVisitor>(mut e: P<Expr>, vis: &mut T) -> Option<P<Expr>> {
    Some({
        vis.visit_expr(&mut e);
        e
    })
}

pub fn noop_flat_map_stmt<T: MutVisitor>(
    Stmt { kind, mut span, mut id }: Stmt,
    vis: &mut T,
) -> SmallVec<[Stmt; 1]> {
    vis.visit_id(&mut id);
    vis.visit_span(&mut span);
    let stmts: SmallVec<_> = noop_flat_map_stmt_kind(kind, vis)
        .into_iter()
        .map(|kind| Stmt { id, kind, span })
        .collect();
    if stmts.len() > 1 {
        panic!(
            "cloning statement `NodeId`s is prohibited by default, \
             the visitor should implement custom statement visiting"
        );
    }
    stmts
}

pub fn noop_flat_map_stmt_kind<T: MutVisitor>(
    kind: StmtKind,
    vis: &mut T,
) -> SmallVec<[StmtKind; 1]> {
    match kind {
        StmtKind::Local(mut local) => smallvec![StmtKind::Local({
            vis.visit_local(&mut local);
            local
        })],
        StmtKind::Item(item) => vis.flat_map_item(item).into_iter().map(StmtKind::Item).collect(),
        StmtKind::Expr(expr) => vis.filter_map_expr(expr).into_iter().map(StmtKind::Expr).collect(),
        StmtKind::Semi(expr) => vis.filter_map_expr(expr).into_iter().map(StmtKind::Semi).collect(),
        StmtKind::Empty => smallvec![StmtKind::Empty],
        StmtKind::MacCall(mut mac) => {
            let MacCallStmt { mac: mac_, style: _, attrs, tokens } = mac.deref_mut();
            vis.visit_mac_call(mac_);
            visit_attrs(attrs, vis);
            visit_lazy_tts(tokens, vis);
            smallvec![StmtKind::MacCall(mac)]
        }
    }
}

pub fn noop_visit_vis<T: MutVisitor>(visibility: &mut Visibility, vis: &mut T) {
    match &mut visibility.kind {
        VisibilityKind::Public | VisibilityKind::Inherited => {}
        VisibilityKind::Restricted { path, id, shorthand: _ } => {
            vis.visit_path(path);
            vis.visit_id(id);
        }
    }
    vis.visit_span(&mut visibility.span);
}

/// Some value for the AST node that is valid but possibly meaningless.
pub trait DummyAstNode {
    fn dummy() -> Self;
}

impl<T> DummyAstNode for Option<T> {
    fn dummy() -> Self {
        Default::default()
    }
}

impl<T: DummyAstNode + 'static> DummyAstNode for P<T> {
    fn dummy() -> Self {
        P(DummyAstNode::dummy())
    }
}

impl DummyAstNode for Item {
    fn dummy() -> Self {
        Item {
            attrs: Default::default(),
            id: DUMMY_NODE_ID,
            span: Default::default(),
            vis: Visibility {
                kind: VisibilityKind::Public,
                span: Default::default(),
                tokens: Default::default(),
            },
            ident: Ident::empty(),
            kind: ItemKind::ExternCrate(None),
            tokens: Default::default(),
        }
    }
}

impl DummyAstNode for Expr {
    fn dummy() -> Self {
        Expr {
            id: DUMMY_NODE_ID,
            kind: ExprKind::Err,
            span: Default::default(),
            attrs: Default::default(),
            tokens: Default::default(),
        }
    }
}

impl DummyAstNode for Ty {
    fn dummy() -> Self {
        Ty {
            id: DUMMY_NODE_ID,
            kind: TyKind::Err,
            span: Default::default(),
            tokens: Default::default(),
        }
    }
}

impl DummyAstNode for Pat {
    fn dummy() -> Self {
        Pat {
            id: DUMMY_NODE_ID,
            kind: PatKind::Wild,
            span: Default::default(),
            tokens: Default::default(),
        }
    }
}

impl DummyAstNode for Stmt {
    fn dummy() -> Self {
        Stmt { id: DUMMY_NODE_ID, kind: StmtKind::Empty, span: Default::default() }
    }
}

impl DummyAstNode for Block {
    fn dummy() -> Self {
        Block {
            stmts: Default::default(),
            id: DUMMY_NODE_ID,
            rules: BlockCheckMode::Default,
            span: Default::default(),
            tokens: Default::default(),
            could_be_bare_literal: Default::default(),
        }
    }
}

impl DummyAstNode for Crate {
    fn dummy() -> Self {
        Crate {
            attrs: Default::default(),
            items: Default::default(),
            spans: Default::default(),
            id: DUMMY_NODE_ID,
            is_placeholder: Default::default(),
        }
    }
}

impl<N: DummyAstNode, T: DummyAstNode> DummyAstNode for crate::ast_traits::AstNodeWrapper<N, T> {
    fn dummy() -> Self {
        crate::ast_traits::AstNodeWrapper::new(N::dummy(), T::dummy())
    }
}
