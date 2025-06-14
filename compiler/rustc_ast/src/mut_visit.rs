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
use rustc_span::source_map::Spanned;
use rustc_span::{Ident, Span};
use smallvec::{Array, SmallVec, smallvec};
use thin_vec::ThinVec;

use crate::ast::*;
use crate::ptr::P;
use crate::tokenstream::*;
use crate::visit::{AssocCtxt, BoundKind, FnCtxt, VisitorResult, try_visit, visit_opt, walk_list};

pub trait ExpectOne<A: Array> {
    fn expect_one(self, err: &'static str) -> A::Item;
}

impl<A: Array> ExpectOne<A> for SmallVec<A> {
    fn expect_one(self, err: &'static str) -> A::Item {
        assert!(self.len() == 1, "{}", err);
        self.into_iter().next().unwrap()
    }
}

mod sealed {
    use rustc_ast_ir::visit::VisitorResult;

    /// This is for compatibility with the regular `Visitor`.
    pub trait MutVisitorResult {
        type Result: VisitorResult;
    }

    impl<T> MutVisitorResult for T {
        type Result = ();
    }
}

use sealed::MutVisitorResult;

pub trait MutVisitor: Sized + MutVisitorResult<Result = ()> {
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
    visit_expr_fields, ExprField, flat_map_expr_field;
    visit_pat_fields, PatField, flat_map_pat_field;
    visit_variants, Variant, flat_map_variant;
    visit_assoc_items, P<AssocItem>, flat_map_assoc_item, ctxt: AssocCtxt;
    visit_where_predicates, WherePredicate, flat_map_where_predicate;
    visit_params, Param, flat_map_param;
    visit_field_defs, FieldDef, flat_map_field_def;
    visit_arms, Arm, flat_map_arm;
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

fn visit_attrs<T: MutVisitor>(vis: &mut T, attrs: &mut AttrVec) {
    for attr in attrs.iter_mut() {
        vis.visit_attribute(attr);
    }
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

pub fn walk_flat_map_arm<T: MutVisitor>(vis: &mut T, mut arm: Arm) -> SmallVec<[Arm; 1]> {
    vis.visit_arm(&mut arm);
    smallvec![arm]
}

pub fn walk_flat_map_variant<T: MutVisitor>(
    vis: &mut T,
    mut variant: Variant,
) -> SmallVec<[Variant; 1]> {
    vis.visit_variant(&mut variant);
    smallvec![variant]
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

pub fn walk_flat_map_param<T: MutVisitor>(vis: &mut T, mut param: Param) -> SmallVec<[Param; 1]> {
    vis.visit_param(&mut param);
    smallvec![param]
}

pub fn walk_flat_map_generic_param<T: MutVisitor>(
    vis: &mut T,
    mut param: GenericParam,
) -> SmallVec<[GenericParam; 1]> {
    vis.visit_generic_param(&mut param);
    smallvec![param]
}

fn walk_ty_alias_where_clauses<T: MutVisitor>(vis: &mut T, tawcs: &mut TyAliasWhereClauses) {
    let TyAliasWhereClauses { before, after, split: _ } = tawcs;
    let TyAliasWhereClause { has_where_token: _, span: span_before } = before;
    let TyAliasWhereClause { has_where_token: _, span: span_after } = after;
    vis.visit_span(span_before);
    vis.visit_span(span_after);
}

pub fn walk_flat_map_where_predicate<T: MutVisitor>(
    vis: &mut T,
    mut pred: WherePredicate,
) -> SmallVec<[WherePredicate; 1]> {
    walk_where_predicate(vis, &mut pred);
    smallvec![pred]
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
