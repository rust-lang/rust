//! A `MutVisitor` represents an AST modification; it accepts an AST piece and
//! mutates it in place. So, for instance, macro expansion is a `MutVisitor`
//! that walks over an AST and modifies it.
//!
//! Note: using a `MutVisitor` (other than the `MacroExpander` `MutVisitor`) on
//! an AST before macro expansion is probably a bad idea. For instance,
//! a `MutVisitor` renaming item names in a module will miss all of those
//! that are created by the expansion of a macro.

use std::ops::DerefMut;
use std::panic;

use rustc_data_structures::flat_map_in_place::FlatMapInPlace;
use rustc_data_structures::stack::ensure_sufficient_stack;
use rustc_span::{Ident, Span};
use smallvec::{Array, SmallVec, smallvec};
use thin_vec::ThinVec;

use crate::ast::*;
use crate::ptr::P;
use crate::tokenstream::*;
use crate::visit::{AssocCtxt, BoundKind, FnCtxt, try_visit, visit_opt, walk_list};

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
    // Methods in this trait have one of three forms:
    //
    //   fn visit_t(&mut self, t: &mut T);                      // common
    //   fn flat_map_t(&mut self, t: T) -> SmallVec<[T; 1]>;    // rare
    //   fn filter_map_t(&mut self, t: T) -> Option<T>;         // rarest
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
        walk_crate(self, c)
    }

    fn visit_meta_list_item(&mut self, list_item: &mut MetaItemInner) {
        walk_meta_list_item(self, list_item);
    }

    fn visit_meta_item(&mut self, meta_item: &mut MetaItem) {
        walk_meta_item(self, meta_item);
    }

    fn visit_use_tree(&mut self, use_tree: &mut UseTree) {
        walk_use_tree(self, use_tree);
    }

    fn visit_foreign_item(&mut self, ni: &mut ForeignItem) {
        walk_item(self, ni);
    }

    fn flat_map_foreign_item(&mut self, ni: P<ForeignItem>) -> SmallVec<[P<ForeignItem>; 1]> {
        walk_flat_map_foreign_item(self, ni)
    }

    fn visit_item(&mut self, i: &mut Item) {
        walk_item(self, i);
    }

    fn flat_map_item(&mut self, i: P<Item>) -> SmallVec<[P<Item>; 1]> {
        walk_flat_map_item(self, i)
    }

    fn visit_fn_header(&mut self, header: &mut FnHeader) {
        walk_fn_header(self, header);
    }

    fn visit_field_def(&mut self, fd: &mut FieldDef) {
        walk_field_def(self, fd);
    }

    fn flat_map_field_def(&mut self, fd: FieldDef) -> SmallVec<[FieldDef; 1]> {
        walk_flat_map_field_def(self, fd)
    }

    fn visit_assoc_item(&mut self, i: &mut AssocItem, ctxt: AssocCtxt) {
        walk_assoc_item(self, i, ctxt)
    }

    fn flat_map_assoc_item(
        &mut self,
        i: P<AssocItem>,
        ctxt: AssocCtxt,
    ) -> SmallVec<[P<AssocItem>; 1]> {
        walk_flat_map_assoc_item(self, i, ctxt)
    }

    fn visit_contract(&mut self, c: &mut FnContract) {
        walk_contract(self, c);
    }

    fn visit_fn_decl(&mut self, d: &mut FnDecl) {
        walk_fn_decl(self, d);
    }

    /// `Span` and `NodeId` are mutated at the caller site.
    fn visit_fn(&mut self, fk: FnKind<'_>, _: Span, _: NodeId) {
        walk_fn(self, fk)
    }

    fn visit_coroutine_kind(&mut self, a: &mut CoroutineKind) {
        walk_coroutine_kind(self, a);
    }

    fn visit_closure_binder(&mut self, b: &mut ClosureBinder) {
        walk_closure_binder(self, b);
    }

    fn visit_block(&mut self, b: &mut Block) {
        walk_block(self, b);
    }

    fn flat_map_stmt(&mut self, s: Stmt) -> SmallVec<[Stmt; 1]> {
        walk_flat_map_stmt(self, s)
    }

    fn visit_arm(&mut self, arm: &mut Arm) {
        walk_arm(self, arm);
    }

    fn flat_map_arm(&mut self, arm: Arm) -> SmallVec<[Arm; 1]> {
        walk_flat_map_arm(self, arm)
    }

    fn visit_pat(&mut self, p: &mut P<Pat>) {
        walk_pat(self, p);
    }

    fn visit_anon_const(&mut self, c: &mut AnonConst) {
        walk_anon_const(self, c);
    }

    fn visit_expr(&mut self, e: &mut P<Expr>) {
        walk_expr(self, e);
    }

    /// This method is a hack to workaround unstable of `stmt_expr_attributes`.
    /// It can be removed once that feature is stabilized.
    fn visit_method_receiver_expr(&mut self, ex: &mut P<Expr>) {
        self.visit_expr(ex)
    }

    fn filter_map_expr(&mut self, e: P<Expr>) -> Option<P<Expr>> {
        walk_filter_map_expr(self, e)
    }

    fn visit_generic_arg(&mut self, arg: &mut GenericArg) {
        walk_generic_arg(self, arg);
    }

    fn visit_ty(&mut self, t: &mut P<Ty>) {
        walk_ty(self, t);
    }

    fn visit_ty_pat(&mut self, t: &mut TyPat) {
        walk_ty_pat(self, t);
    }

    fn visit_lifetime(&mut self, l: &mut Lifetime) {
        walk_lifetime(self, l);
    }

    fn visit_assoc_item_constraint(&mut self, c: &mut AssocItemConstraint) {
        walk_assoc_item_constraint(self, c);
    }

    fn visit_foreign_mod(&mut self, nm: &mut ForeignMod) {
        walk_foreign_mod(self, nm);
    }

    fn visit_variant(&mut self, v: &mut Variant) {
        walk_variant(self, v);
    }

    fn flat_map_variant(&mut self, v: Variant) -> SmallVec<[Variant; 1]> {
        walk_flat_map_variant(self, v)
    }

    fn visit_ident(&mut self, i: &mut Ident) {
        self.visit_span(&mut i.span);
    }

    fn visit_path(&mut self, p: &mut Path) {
        walk_path(self, p);
    }

    fn visit_path_segment(&mut self, p: &mut PathSegment) {
        walk_path_segment(self, p)
    }

    fn visit_qself(&mut self, qs: &mut Option<P<QSelf>>) {
        walk_qself(self, qs);
    }

    fn visit_generic_args(&mut self, p: &mut GenericArgs) {
        walk_generic_args(self, p);
    }

    fn visit_angle_bracketed_parameter_data(&mut self, p: &mut AngleBracketedArgs) {
        walk_angle_bracketed_parameter_data(self, p);
    }

    fn visit_parenthesized_parameter_data(&mut self, p: &mut ParenthesizedArgs) {
        walk_parenthesized_parameter_data(self, p);
    }

    fn visit_local(&mut self, l: &mut Local) {
        walk_local(self, l);
    }

    fn visit_mac_call(&mut self, mac: &mut MacCall) {
        walk_mac(self, mac);
    }

    fn visit_macro_def(&mut self, def: &mut MacroDef) {
        walk_macro_def(self, def);
    }

    fn visit_label(&mut self, label: &mut Label) {
        walk_label(self, label);
    }

    fn visit_attribute(&mut self, at: &mut Attribute) {
        walk_attribute(self, at);
    }

    fn visit_param(&mut self, param: &mut Param) {
        walk_param(self, param);
    }

    fn flat_map_param(&mut self, param: Param) -> SmallVec<[Param; 1]> {
        walk_flat_map_param(self, param)
    }

    fn visit_generics(&mut self, generics: &mut Generics) {
        walk_generics(self, generics);
    }

    fn visit_trait_ref(&mut self, tr: &mut TraitRef) {
        walk_trait_ref(self, tr);
    }

    fn visit_poly_trait_ref(&mut self, p: &mut PolyTraitRef) {
        walk_poly_trait_ref(self, p);
    }

    fn visit_variant_data(&mut self, vdata: &mut VariantData) {
        walk_variant_data(self, vdata);
    }

    fn visit_generic_param(&mut self, param: &mut GenericParam) {
        walk_generic_param(self, param)
    }

    fn flat_map_generic_param(&mut self, param: GenericParam) -> SmallVec<[GenericParam; 1]> {
        walk_flat_map_generic_param(self, param)
    }

    fn visit_param_bound(&mut self, tpb: &mut GenericBound, _ctxt: BoundKind) {
        walk_param_bound(self, tpb);
    }

    fn visit_precise_capturing_arg(&mut self, arg: &mut PreciseCapturingArg) {
        walk_precise_capturing_arg(self, arg);
    }

    fn visit_expr_field(&mut self, f: &mut ExprField) {
        walk_expr_field(self, f);
    }

    fn flat_map_expr_field(&mut self, f: ExprField) -> SmallVec<[ExprField; 1]> {
        walk_flat_map_expr_field(self, f)
    }

    fn visit_where_clause(&mut self, where_clause: &mut WhereClause) {
        walk_where_clause(self, where_clause);
    }

    fn flat_map_where_predicate(
        &mut self,
        where_predicate: WherePredicate,
    ) -> SmallVec<[WherePredicate; 1]> {
        walk_flat_map_where_predicate(self, where_predicate)
    }

    fn visit_where_predicate_kind(&mut self, kind: &mut WherePredicateKind) {
        walk_where_predicate_kind(self, kind)
    }

    fn visit_vis(&mut self, vis: &mut Visibility) {
        walk_vis(self, vis);
    }

    fn visit_id(&mut self, _id: &mut NodeId) {
        // Do nothing.
    }

    // Span visiting is no longer used, but we keep it for now,
    // in case it's needed for something like #127241.
    fn visit_span(&mut self, _sp: &mut Span) {
        // Do nothing.
    }

    fn visit_pat_field(&mut self, fp: &mut PatField) {
        walk_pat_field(self, fp)
    }

    fn flat_map_pat_field(&mut self, fp: PatField) -> SmallVec<[PatField; 1]> {
        walk_flat_map_pat_field(self, fp)
    }

    fn visit_inline_asm(&mut self, asm: &mut InlineAsm) {
        walk_inline_asm(self, asm)
    }

    fn visit_inline_asm_sym(&mut self, sym: &mut InlineAsmSym) {
        walk_inline_asm_sym(self, sym)
    }

    fn visit_format_args(&mut self, fmt: &mut FormatArgs) {
        walk_format_args(self, fmt)
    }

    fn visit_capture_by(&mut self, capture_by: &mut CaptureBy) {
        walk_capture_by(self, capture_by)
    }

    fn visit_fn_ret_ty(&mut self, fn_ret_ty: &mut FnRetTy) {
        walk_fn_ret_ty(self, fn_ret_ty)
    }
}

super::common_visitor_and_walkers!((mut) MutVisitor);

macro_rules! generate_flat_map_visitor_fns {
    ($($name:ident, $Ty:ty, $flat_map_fn:ident$(, $param:ident: $ParamTy:ty)*;)+) => {
        $(
            fn $name<V: MutVisitor>(
                vis: &mut V,
                values: &mut ThinVec<$Ty>,
                $(
                    $param: $ParamTy,
                )*
            ) {
                values.flat_map_in_place(|value| vis.$flat_map_fn(value$(,$param)*));
            }
        )+
    }
}

generate_flat_map_visitor_fns! {
    visit_items, P<Item>, flat_map_item;
    visit_foreign_items, P<ForeignItem>, flat_map_foreign_item;
    visit_generic_params, GenericParam, flat_map_generic_param;
    visit_stmts, Stmt, flat_map_stmt;
    visit_exprs, P<Expr>, filter_map_expr;
    visit_pat_fields, PatField, flat_map_pat_field;
    visit_variants, Variant, flat_map_variant;
    visit_assoc_items, P<AssocItem>, flat_map_assoc_item, ctxt: AssocCtxt;
}

#[inline]
fn visit_vec<T, F>(elems: &mut Vec<T>, mut visit_elem: F)
where
    F: FnMut(&mut T),
{
    for elem in elems {
        visit_elem(elem);
    }
}

#[inline]
fn visit_thin_vec<T, F>(elems: &mut ThinVec<T>, mut visit_elem: F)
where
    F: FnMut(&mut T),
{
    for elem in elems {
        visit_elem(elem);
    }
}

#[inline]
fn visit_opt<T, F>(opt: &mut Option<T>, mut visit_elem: F)
where
    F: FnMut(&mut T),
{
    if let Some(elem) = opt {
        visit_elem(elem);
    }
}

fn visit_attrs<T: MutVisitor>(vis: &mut T, attrs: &mut AttrVec) {
    for attr in attrs.iter_mut() {
        vis.visit_attribute(attr);
    }
}

fn visit_attr_args<T: MutVisitor>(vis: &mut T, args: &mut AttrArgs) {
    match args {
        AttrArgs::Empty => {}
        AttrArgs::Delimited(args) => visit_delim_args(vis, args),
        AttrArgs::Eq { eq_span, expr } => {
            vis.visit_expr(expr);
            vis.visit_span(eq_span);
        }
    }
}

fn visit_delim_args<T: MutVisitor>(vis: &mut T, args: &mut DelimArgs) {
    let DelimArgs { dspan, delim: _, tokens: _ } = args;
    let DelimSpan { open, close } = dspan;
    vis.visit_span(open);
    vis.visit_span(close);
}

pub fn walk_flat_map_pat_field<T: MutVisitor>(
    vis: &mut T,
    mut fp: PatField,
) -> SmallVec<[PatField; 1]> {
    vis.visit_pat_field(&mut fp);
    smallvec![fp]
}

fn visit_nested_use_tree<V: MutVisitor>(
    vis: &mut V,
    nested_tree: &mut UseTree,
    nested_id: &mut NodeId,
) {
    vis.visit_id(nested_id);
    vis.visit_use_tree(nested_tree);
}

pub fn walk_arm<T: MutVisitor>(vis: &mut T, arm: &mut Arm) {
    let Arm { attrs, pat, guard, body, span, id, is_placeholder: _ } = arm;
    vis.visit_id(id);
    visit_attrs(vis, attrs);
    vis.visit_pat(pat);
    visit_opt(guard, |guard| vis.visit_expr(guard));
    visit_opt(body, |body| vis.visit_expr(body));
    vis.visit_span(span);
}

pub fn walk_flat_map_arm<T: MutVisitor>(vis: &mut T, mut arm: Arm) -> SmallVec<[Arm; 1]> {
    vis.visit_arm(&mut arm);
    smallvec![arm]
}

fn walk_assoc_item_constraint<T: MutVisitor>(
    vis: &mut T,
    AssocItemConstraint { id, ident, gen_args, kind, span }: &mut AssocItemConstraint,
) {
    vis.visit_id(id);
    vis.visit_ident(ident);
    if let Some(gen_args) = gen_args {
        vis.visit_generic_args(gen_args);
    }
    match kind {
        AssocItemConstraintKind::Equality { term } => match term {
            Term::Ty(ty) => vis.visit_ty(ty),
            Term::Const(c) => vis.visit_anon_const(c),
        },
        AssocItemConstraintKind::Bound { bounds } => visit_bounds(vis, bounds, BoundKind::Bound),
    }
    vis.visit_span(span);
}

pub fn walk_flat_map_variant<T: MutVisitor>(
    vis: &mut T,
    mut variant: Variant,
) -> SmallVec<[Variant; 1]> {
    vis.visit_variant(&mut variant);
    smallvec![variant]
}

fn walk_generic_args<T: MutVisitor>(vis: &mut T, generic_args: &mut GenericArgs) {
    match generic_args {
        GenericArgs::AngleBracketed(data) => vis.visit_angle_bracketed_parameter_data(data),
        GenericArgs::Parenthesized(data) => vis.visit_parenthesized_parameter_data(data),
        GenericArgs::ParenthesizedElided(span) => vis.visit_span(span),
    }
}

fn walk_generic_arg<T: MutVisitor>(vis: &mut T, arg: &mut GenericArg) {
    match arg {
        GenericArg::Lifetime(lt) => vis.visit_lifetime(lt),
        GenericArg::Type(ty) => vis.visit_ty(ty),
        GenericArg::Const(ct) => vis.visit_anon_const(ct),
    }
}

fn walk_angle_bracketed_parameter_data<T: MutVisitor>(vis: &mut T, data: &mut AngleBracketedArgs) {
    let AngleBracketedArgs { args, span } = data;
    visit_thin_vec(args, |arg| match arg {
        AngleBracketedArg::Arg(arg) => vis.visit_generic_arg(arg),
        AngleBracketedArg::Constraint(constraint) => vis.visit_assoc_item_constraint(constraint),
    });
    vis.visit_span(span);
}

fn walk_parenthesized_parameter_data<T: MutVisitor>(vis: &mut T, args: &mut ParenthesizedArgs) {
    let ParenthesizedArgs { inputs, output, span, inputs_span } = args;
    visit_thin_vec(inputs, |input| vis.visit_ty(input));
    vis.visit_fn_ret_ty(output);
    vis.visit_span(span);
    vis.visit_span(inputs_span);
}

fn walk_attribute<T: MutVisitor>(vis: &mut T, attr: &mut Attribute) {
    let Attribute { kind, id: _, style: _, span } = attr;
    match kind {
        AttrKind::Normal(normal) => {
            let NormalAttr { item: AttrItem { unsafety: _, path, args, tokens: _ }, tokens: _ } =
                &mut **normal;
            vis.visit_path(path);
            visit_attr_args(vis, args);
        }
        AttrKind::DocComment(_kind, _sym) => {}
    }
    vis.visit_span(span);
}

fn walk_mac<T: MutVisitor>(vis: &mut T, mac: &mut MacCall) {
    let MacCall { path, args } = mac;
    vis.visit_path(path);
    visit_delim_args(vis, args);
}

fn walk_macro_def<T: MutVisitor>(vis: &mut T, macro_def: &mut MacroDef) {
    let MacroDef { body, macro_rules: _ } = macro_def;
    visit_delim_args(vis, body);
}

fn walk_meta_list_item<T: MutVisitor>(vis: &mut T, li: &mut MetaItemInner) {
    match li {
        MetaItemInner::MetaItem(mi) => vis.visit_meta_item(mi),
        MetaItemInner::Lit(_lit) => {}
    }
}

fn walk_meta_item<T: MutVisitor>(vis: &mut T, mi: &mut MetaItem) {
    let MetaItem { unsafety: _, path: _, kind, span } = mi;
    match kind {
        MetaItemKind::Word => {}
        MetaItemKind::List(mis) => visit_thin_vec(mis, |mi| vis.visit_meta_list_item(mi)),
        MetaItemKind::NameValue(_s) => {}
    }
    vis.visit_span(span);
}

pub fn walk_param<T: MutVisitor>(vis: &mut T, param: &mut Param) {
    let Param { attrs, id, pat, span, ty, is_placeholder: _ } = param;
    vis.visit_id(id);
    visit_attrs(vis, attrs);
    vis.visit_pat(pat);
    vis.visit_ty(ty);
    vis.visit_span(span);
}

pub fn walk_flat_map_param<T: MutVisitor>(vis: &mut T, mut param: Param) -> SmallVec<[Param; 1]> {
    vis.visit_param(&mut param);
    smallvec![param]
}

fn walk_closure_binder<T: MutVisitor>(vis: &mut T, binder: &mut ClosureBinder) {
    match binder {
        ClosureBinder::NotPresent => {}
        ClosureBinder::For { span: _, generic_params } => {
            generic_params.flat_map_in_place(|param| vis.flat_map_generic_param(param));
        }
    }
}

fn walk_fn<T: MutVisitor>(vis: &mut T, kind: FnKind<'_>) {
    match kind {
        FnKind::Fn(
            _ctxt,
            _vis,
            Fn {
                defaultness,
                ident,
                generics,
                contract,
                body,
                sig: FnSig { header, decl, span },
                define_opaque,
            },
        ) => {
            // Visibility is visited as a part of the item.
            visit_defaultness(vis, defaultness);
            vis.visit_ident(ident);
            vis.visit_fn_header(header);
            vis.visit_generics(generics);
            vis.visit_fn_decl(decl);
            if let Some(contract) = contract {
                vis.visit_contract(contract);
            }
            if let Some(body) = body {
                vis.visit_block(body);
            }
            vis.visit_span(span);

            walk_define_opaques(vis, define_opaque);
        }
        FnKind::Closure(binder, coroutine_kind, decl, body) => {
            vis.visit_closure_binder(binder);
            coroutine_kind.as_mut().map(|coroutine_kind| vis.visit_coroutine_kind(coroutine_kind));
            vis.visit_fn_decl(decl);
            vis.visit_expr(body);
        }
    }
}

fn walk_contract<T: MutVisitor>(vis: &mut T, contract: &mut FnContract) {
    let FnContract { requires, ensures } = contract;
    if let Some(pred) = requires {
        vis.visit_expr(pred);
    }
    if let Some(pred) = ensures {
        vis.visit_expr(pred);
    }
}

fn walk_fn_decl<T: MutVisitor>(vis: &mut T, decl: &mut FnDecl) {
    let FnDecl { inputs, output } = decl;
    inputs.flat_map_in_place(|param| vis.flat_map_param(param));
    vis.visit_fn_ret_ty(output);
}

fn walk_fn_ret_ty<T: MutVisitor>(vis: &mut T, fn_ret_ty: &mut FnRetTy) {
    match fn_ret_ty {
        FnRetTy::Default(span) => vis.visit_span(span),
        FnRetTy::Ty(ty) => vis.visit_ty(ty),
    }
}

fn walk_param_bound<T: MutVisitor>(vis: &mut T, pb: &mut GenericBound) {
    match pb {
        GenericBound::Trait(trait_ref) => vis.visit_poly_trait_ref(trait_ref),
        GenericBound::Outlives(lifetime) => walk_lifetime(vis, lifetime),
        GenericBound::Use(args, span) => {
            for arg in args {
                vis.visit_precise_capturing_arg(arg);
            }
            vis.visit_span(span);
        }
    }
}

fn walk_precise_capturing_arg<T: MutVisitor>(vis: &mut T, arg: &mut PreciseCapturingArg) {
    match arg {
        PreciseCapturingArg::Lifetime(lt) => {
            vis.visit_lifetime(lt);
        }
        PreciseCapturingArg::Arg(path, id) => {
            vis.visit_id(id);
            vis.visit_path(path);
        }
    }
}

pub fn walk_generic_param<T: MutVisitor>(vis: &mut T, param: &mut GenericParam) {
    let GenericParam { id, ident, attrs, bounds, kind, colon_span, is_placeholder: _ } = param;
    vis.visit_id(id);
    visit_attrs(vis, attrs);
    vis.visit_ident(ident);
    visit_vec(bounds, |bound| vis.visit_param_bound(bound, BoundKind::Bound));
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
    if let Some(colon_span) = colon_span {
        vis.visit_span(colon_span);
    }
}

pub fn walk_flat_map_generic_param<T: MutVisitor>(
    vis: &mut T,
    mut param: GenericParam,
) -> SmallVec<[GenericParam; 1]> {
    vis.visit_generic_param(&mut param);
    smallvec![param]
}

fn walk_generics<T: MutVisitor>(vis: &mut T, generics: &mut Generics) {
    let Generics { params, where_clause, span } = generics;
    params.flat_map_in_place(|param| vis.flat_map_generic_param(param));
    vis.visit_where_clause(where_clause);
    vis.visit_span(span);
}

fn walk_ty_alias_where_clauses<T: MutVisitor>(vis: &mut T, tawcs: &mut TyAliasWhereClauses) {
    let TyAliasWhereClauses { before, after, split: _ } = tawcs;
    let TyAliasWhereClause { has_where_token: _, span: span_before } = before;
    let TyAliasWhereClause { has_where_token: _, span: span_after } = after;
    vis.visit_span(span_before);
    vis.visit_span(span_after);
}

fn walk_where_clause<T: MutVisitor>(vis: &mut T, wc: &mut WhereClause) {
    let WhereClause { has_where_token: _, predicates, span } = wc;
    predicates.flat_map_in_place(|predicate| vis.flat_map_where_predicate(predicate));
    vis.visit_span(span);
}

pub fn walk_flat_map_where_predicate<T: MutVisitor>(
    vis: &mut T,
    mut pred: WherePredicate,
) -> SmallVec<[WherePredicate; 1]> {
    let WherePredicate { attrs, kind, id, span, is_placeholder: _ } = &mut pred;
    vis.visit_id(id);
    visit_attrs(vis, attrs);
    vis.visit_where_predicate_kind(kind);
    vis.visit_span(span);
    smallvec![pred]
}

pub fn walk_where_predicate_kind<T: MutVisitor>(vis: &mut T, kind: &mut WherePredicateKind) {
    match kind {
        WherePredicateKind::BoundPredicate(bp) => {
            let WhereBoundPredicate { bound_generic_params, bounded_ty, bounds } = bp;
            bound_generic_params.flat_map_in_place(|param| vis.flat_map_generic_param(param));
            vis.visit_ty(bounded_ty);
            visit_vec(bounds, |bound| vis.visit_param_bound(bound, BoundKind::Bound));
        }
        WherePredicateKind::RegionPredicate(rp) => {
            let WhereRegionPredicate { lifetime, bounds } = rp;
            vis.visit_lifetime(lifetime);
            visit_vec(bounds, |bound| vis.visit_param_bound(bound, BoundKind::Bound));
        }
        WherePredicateKind::EqPredicate(ep) => {
            let WhereEqPredicate { lhs_ty, rhs_ty } = ep;
            vis.visit_ty(lhs_ty);
            vis.visit_ty(rhs_ty);
        }
    }
}

fn walk_variant_data<T: MutVisitor>(vis: &mut T, vdata: &mut VariantData) {
    match vdata {
        VariantData::Struct { fields, recovered: _ } => {
            fields.flat_map_in_place(|field| vis.flat_map_field_def(field));
        }
        VariantData::Tuple(fields, id) => {
            vis.visit_id(id);
            fields.flat_map_in_place(|field| vis.flat_map_field_def(field));
        }
        VariantData::Unit(id) => vis.visit_id(id),
    }
}

pub fn walk_field_def<T: MutVisitor>(visitor: &mut T, fd: &mut FieldDef) {
    let FieldDef { span, ident, vis, id, ty, attrs, is_placeholder: _, safety, default } = fd;
    visitor.visit_id(id);
    visit_attrs(visitor, attrs);
    visitor.visit_vis(vis);
    visit_safety(visitor, safety);
    visit_opt(ident, |ident| visitor.visit_ident(ident));
    visitor.visit_ty(ty);
    visit_opt(default, |default| visitor.visit_anon_const(default));
    visitor.visit_span(span);
}

pub fn walk_flat_map_field_def<T: MutVisitor>(
    vis: &mut T,
    mut fd: FieldDef,
) -> SmallVec<[FieldDef; 1]> {
    vis.visit_field_def(&mut fd);
    smallvec![fd]
}

pub fn walk_flat_map_expr_field<T: MutVisitor>(
    vis: &mut T,
    mut f: ExprField,
) -> SmallVec<[ExprField; 1]> {
    vis.visit_expr_field(&mut f);
    smallvec![f]
}

pub fn walk_item_kind<K: WalkItemKind>(
    kind: &mut K,
    span: Span,
    id: NodeId,
    visibility: &mut Visibility,
    ctxt: K::Ctxt,
    vis: &mut impl MutVisitor,
) {
    kind.walk(span, id, visibility, ctxt, vis)
}

pub fn walk_flat_map_item(vis: &mut impl MutVisitor, mut item: P<Item>) -> SmallVec<[P<Item>; 1]> {
    vis.visit_item(&mut item);
    smallvec![item]
}

pub fn walk_flat_map_foreign_item(
    vis: &mut impl MutVisitor,
    mut item: P<ForeignItem>,
) -> SmallVec<[P<ForeignItem>; 1]> {
    vis.visit_foreign_item(&mut item);
    smallvec![item]
}

pub fn walk_flat_map_assoc_item(
    vis: &mut impl MutVisitor,
    mut item: P<AssocItem>,
    ctxt: AssocCtxt,
) -> SmallVec<[P<AssocItem>; 1]> {
    vis.visit_assoc_item(&mut item, ctxt);
    smallvec![item]
}

fn walk_inline_asm<T: MutVisitor>(vis: &mut T, asm: &mut InlineAsm) {
    // FIXME: Visit spans inside all this currently ignored stuff.
    let InlineAsm {
        asm_macro: _,
        template: _,
        template_strs: _,
        operands,
        clobber_abis: _,
        options: _,
        line_spans: _,
    } = asm;
    for (op, span) in operands {
        match op {
            InlineAsmOperand::In { expr, reg: _ }
            | InlineAsmOperand::Out { expr: Some(expr), reg: _, late: _ }
            | InlineAsmOperand::InOut { expr, reg: _, late: _ } => vis.visit_expr(expr),
            InlineAsmOperand::Out { expr: None, reg: _, late: _ } => {}
            InlineAsmOperand::SplitInOut { in_expr, out_expr, reg: _, late: _ } => {
                vis.visit_expr(in_expr);
                if let Some(out_expr) = out_expr {
                    vis.visit_expr(out_expr);
                }
            }
            InlineAsmOperand::Const { anon_const } => vis.visit_anon_const(anon_const),
            InlineAsmOperand::Sym { sym } => vis.visit_inline_asm_sym(sym),
            InlineAsmOperand::Label { block } => vis.visit_block(block),
        }
        vis.visit_span(span);
    }
}

fn walk_inline_asm_sym<T: MutVisitor>(
    vis: &mut T,
    InlineAsmSym { id, qself, path }: &mut InlineAsmSym,
) {
    vis.visit_id(id);
    vis.visit_qself(qself);
    vis.visit_path(path);
}

fn walk_format_args<T: MutVisitor>(vis: &mut T, fmt: &mut FormatArgs) {
    // FIXME: visit the template exhaustively.
    let FormatArgs { span, template: _, arguments, uncooked_fmt_str: _ } = fmt;
    for FormatArgument { kind, expr } in arguments.all_args_mut() {
        match kind {
            FormatArgumentKind::Named(ident) | FormatArgumentKind::Captured(ident) => {
                vis.visit_ident(ident)
            }
            FormatArgumentKind::Normal => {}
        }
        vis.visit_expr(expr);
    }
    vis.visit_span(span);
}

pub fn walk_expr<T: MutVisitor>(vis: &mut T, Expr { kind, id, span, attrs, tokens: _ }: &mut Expr) {
    vis.visit_id(id);
    visit_attrs(vis, attrs);
    match kind {
        ExprKind::Array(exprs) => visit_exprs(vis, exprs),
        ExprKind::ConstBlock(anon_const) => {
            vis.visit_anon_const(anon_const);
        }
        ExprKind::Repeat(expr, count) => {
            vis.visit_expr(expr);
            vis.visit_anon_const(count);
        }
        ExprKind::Tup(exprs) => visit_exprs(vis, exprs),
        ExprKind::Call(f, args) => {
            vis.visit_expr(f);
            visit_exprs(vis, args);
        }
        ExprKind::MethodCall(box MethodCall {
            seg: PathSegment { ident, id, args: seg_args },
            receiver,
            args: call_args,
            span,
        }) => {
            vis.visit_method_receiver_expr(receiver);
            vis.visit_id(id);
            vis.visit_ident(ident);
            visit_opt(seg_args, |args| vis.visit_generic_args(args));
            visit_exprs(vis, call_args);
            vis.visit_span(span);
        }
        ExprKind::Binary(binop, lhs, rhs) => {
            vis.visit_expr(lhs);
            vis.visit_expr(rhs);
            vis.visit_span(&mut binop.span);
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
        ExprKind::AddrOf(_kind, _mut, ohs) => vis.visit_expr(ohs),
        ExprKind::Let(pat, scrutinee, span, _recovered) => {
            vis.visit_pat(pat);
            vis.visit_expr(scrutinee);
            vis.visit_span(span);
        }
        ExprKind::If(cond, tr, fl) => {
            vis.visit_expr(cond);
            vis.visit_block(tr);
            visit_opt(fl, |fl| ensure_sufficient_stack(|| vis.visit_expr(fl)));
        }
        ExprKind::While(cond, body, label) => {
            visit_opt(label, |label| vis.visit_label(label));
            vis.visit_expr(cond);
            vis.visit_block(body);
        }
        ExprKind::ForLoop { pat, iter, body, label, kind: _ } => {
            visit_opt(label, |label| vis.visit_label(label));
            vis.visit_pat(pat);
            vis.visit_expr(iter);
            vis.visit_block(body);
        }
        ExprKind::Loop(body, label, span) => {
            visit_opt(label, |label| vis.visit_label(label));
            vis.visit_block(body);
            vis.visit_span(span);
        }
        ExprKind::Match(expr, arms, _kind) => {
            vis.visit_expr(expr);
            arms.flat_map_in_place(|arm| vis.flat_map_arm(arm));
        }
        ExprKind::Closure(box Closure {
            binder,
            capture_clause,
            constness,
            coroutine_kind,
            movability: _,
            fn_decl,
            body,
            fn_decl_span,
            fn_arg_span,
        }) => {
            visit_constness(vis, constness);
            vis.visit_capture_by(capture_clause);
            vis.visit_fn(FnKind::Closure(binder, coroutine_kind, fn_decl, body), *span, *id);
            vis.visit_span(fn_decl_span);
            vis.visit_span(fn_arg_span);
        }
        ExprKind::Block(blk, label) => {
            visit_opt(label, |label| vis.visit_label(label));
            vis.visit_block(blk);
        }
        ExprKind::Gen(_capture_by, body, _kind, decl_span) => {
            vis.visit_block(body);
            vis.visit_span(decl_span);
        }
        ExprKind::Await(expr, await_kw_span) => {
            vis.visit_expr(expr);
            vis.visit_span(await_kw_span);
        }
        ExprKind::Use(expr, use_kw_span) => {
            vis.visit_expr(expr);
            vis.visit_span(use_kw_span);
        }
        ExprKind::Assign(el, er, span) => {
            vis.visit_expr(el);
            vis.visit_expr(er);
            vis.visit_span(span);
        }
        ExprKind::AssignOp(_op, el, er) => {
            vis.visit_expr(el);
            vis.visit_expr(er);
        }
        ExprKind::Field(el, ident) => {
            vis.visit_expr(el);
            vis.visit_ident(ident);
        }
        ExprKind::Index(el, er, brackets_span) => {
            vis.visit_expr(el);
            vis.visit_expr(er);
            vis.visit_span(brackets_span);
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
        ExprKind::Become(expr) => vis.visit_expr(expr),
        ExprKind::InlineAsm(asm) => vis.visit_inline_asm(asm),
        ExprKind::FormatArgs(fmt) => vis.visit_format_args(fmt),
        ExprKind::OffsetOf(container, fields) => {
            vis.visit_ty(container);
            for field in fields.iter_mut() {
                vis.visit_ident(field);
            }
        }
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
        ExprKind::Yield(kind) => {
            let expr = kind.expr_mut();
            if let Some(expr) = expr {
                vis.visit_expr(expr);
            }
        }
        ExprKind::Try(expr) => vis.visit_expr(expr),
        ExprKind::TryBlock(body) => vis.visit_block(body),
        ExprKind::Lit(_token) => {}
        ExprKind::IncludedBytes(_bytes) => {}
        ExprKind::UnsafeBinderCast(_kind, expr, ty) => {
            vis.visit_expr(expr);
            if let Some(ty) = ty {
                vis.visit_ty(ty);
            }
        }
        ExprKind::Err(_guar) => {}
        ExprKind::Dummy => {}
    }
    vis.visit_span(span);
}

pub fn walk_filter_map_expr<T: MutVisitor>(vis: &mut T, mut e: P<Expr>) -> Option<P<Expr>> {
    vis.visit_expr(&mut e);
    Some(e)
}

pub fn walk_flat_map_stmt<T: MutVisitor>(
    vis: &mut T,
    Stmt { kind, span, mut id }: Stmt,
) -> SmallVec<[Stmt; 1]> {
    vis.visit_id(&mut id);
    let mut stmts: SmallVec<[Stmt; 1]> = walk_flat_map_stmt_kind(vis, kind)
        .into_iter()
        .map(|kind| Stmt { id, kind, span })
        .collect();
    match &mut stmts[..] {
        [] => {}
        [stmt] => vis.visit_span(&mut stmt.span),
        _ => panic!(
            "cloning statement `NodeId`s is prohibited by default, \
             the visitor should implement custom statement visiting"
        ),
    }
    stmts
}

fn walk_flat_map_stmt_kind<T: MutVisitor>(vis: &mut T, kind: StmtKind) -> SmallVec<[StmtKind; 1]> {
    match kind {
        StmtKind::Let(mut local) => smallvec![StmtKind::Let({
            vis.visit_local(&mut local);
            local
        })],
        StmtKind::Item(item) => vis.flat_map_item(item).into_iter().map(StmtKind::Item).collect(),
        StmtKind::Expr(expr) => vis.filter_map_expr(expr).into_iter().map(StmtKind::Expr).collect(),
        StmtKind::Semi(expr) => vis.filter_map_expr(expr).into_iter().map(StmtKind::Semi).collect(),
        StmtKind::Empty => smallvec![StmtKind::Empty],
        StmtKind::MacCall(mut mac) => {
            let MacCallStmt { mac: mac_, style: _, attrs, tokens: _ } = mac.deref_mut();
            visit_attrs(vis, attrs);
            vis.visit_mac_call(mac_);
            smallvec![StmtKind::MacCall(mac)]
        }
    }
}

fn walk_vis<T: MutVisitor>(vis: &mut T, visibility: &mut Visibility) {
    let Visibility { kind, span, tokens: _ } = visibility;
    match kind {
        VisibilityKind::Public | VisibilityKind::Inherited => {}
        VisibilityKind::Restricted { path, id, shorthand: _ } => {
            vis.visit_id(id);
            vis.visit_path(path);
        }
    }
    vis.visit_span(span);
}

fn walk_capture_by<T: MutVisitor>(vis: &mut T, capture_by: &mut CaptureBy) {
    match capture_by {
        CaptureBy::Ref => {}
        CaptureBy::Value { move_kw } => {
            vis.visit_span(move_kw);
        }
        CaptureBy::Use { use_kw } => {
            vis.visit_span(use_kw);
        }
    }
}

#[derive(Debug)]
pub enum FnKind<'a> {
    /// E.g., `fn foo()`, `fn foo(&self)`, or `extern "Abi" fn foo()`.
    Fn(FnCtxt, &'a mut Visibility, &'a mut Fn),

    /// E.g., `|x, y| body`.
    Closure(
        &'a mut ClosureBinder,
        &'a mut Option<CoroutineKind>,
        &'a mut P<FnDecl>,
        &'a mut P<Expr>,
    ),
}
