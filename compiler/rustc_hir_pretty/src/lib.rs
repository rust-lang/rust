//! HIR pretty-printing is layered on top of AST pretty-printing. A number of
//! the definitions in this file have equivalents in `rustc_ast_pretty`.

// tidy-alphabetical-start
#![recursion_limit = "256"]
// tidy-alphabetical-end

use std::cell::Cell;
use std::vec;

use rustc_abi::ExternAbi;
use rustc_ast::util::parser::{self, ExprPrecedence, Fixity};
use rustc_ast::{AttrStyle, DUMMY_NODE_ID, DelimArgs};
use rustc_ast_pretty::pp::Breaks::{Consistent, Inconsistent};
use rustc_ast_pretty::pp::{self, BoxMarker, Breaks};
use rustc_ast_pretty::pprust::state::MacHeader;
use rustc_ast_pretty::pprust::{Comments, PrintState};
use rustc_attr_data_structures::{AttributeKind, PrintAttribute};
use rustc_hir::{
    BindingMode, ByRef, ConstArgKind, GenericArg, GenericBound, GenericParam, GenericParamKind,
    HirId, ImplicitSelfKind, LifetimeParamKind, Node, PatKind, PreciseCapturingArg, RangeEnd, Term,
    TyPatKind,
};
use rustc_span::source_map::SourceMap;
use rustc_span::{FileName, Ident, Span, Symbol, kw, sym};
use {rustc_ast as ast, rustc_hir as hir};

pub fn id_to_string(cx: &dyn rustc_hir::intravisit::HirTyCtxt<'_>, hir_id: HirId) -> String {
    to_string(&cx, |s| s.print_node(cx.hir_node(hir_id)))
}

pub enum AnnNode<'a> {
    Name(&'a Symbol),
    Block(&'a hir::Block<'a>),
    Item(&'a hir::Item<'a>),
    SubItem(HirId),
    Expr(&'a hir::Expr<'a>),
    Pat(&'a hir::Pat<'a>),
    TyPat(&'a hir::TyPat<'a>),
    Arm(&'a hir::Arm<'a>),
}

pub enum Nested {
    Item(hir::ItemId),
    TraitItem(hir::TraitItemId),
    ImplItem(hir::ImplItemId),
    ForeignItem(hir::ForeignItemId),
    Body(hir::BodyId),
    BodyParamPat(hir::BodyId, usize),
}

pub trait PpAnn {
    fn nested(&self, _state: &mut State<'_>, _nested: Nested) {}
    fn pre(&self, _state: &mut State<'_>, _node: AnnNode<'_>) {}
    fn post(&self, _state: &mut State<'_>, _node: AnnNode<'_>) {}
}

impl PpAnn for &dyn rustc_hir::intravisit::HirTyCtxt<'_> {
    fn nested(&self, state: &mut State<'_>, nested: Nested) {
        match nested {
            Nested::Item(id) => state.print_item(self.hir_item(id)),
            Nested::TraitItem(id) => state.print_trait_item(self.hir_trait_item(id)),
            Nested::ImplItem(id) => state.print_impl_item(self.hir_impl_item(id)),
            Nested::ForeignItem(id) => state.print_foreign_item(self.hir_foreign_item(id)),
            Nested::Body(id) => state.print_expr(self.hir_body(id).value),
            Nested::BodyParamPat(id, i) => state.print_pat(self.hir_body(id).params[i].pat),
        }
    }
}

pub struct State<'a> {
    pub s: pp::Printer,
    comments: Option<Comments<'a>>,
    attrs: &'a dyn Fn(HirId) -> &'a [hir::Attribute],
    ann: &'a (dyn PpAnn + 'a),
}

impl<'a> State<'a> {
    fn attrs(&self, id: HirId) -> &'a [hir::Attribute] {
        (self.attrs)(id)
    }

    fn precedence(&self, expr: &hir::Expr<'_>) -> ExprPrecedence {
        let for_each_attr = |id: HirId, callback: &mut dyn FnMut(&hir::Attribute)| {
            self.attrs(id).iter().for_each(callback);
        };
        expr.precedence(&for_each_attr)
    }

    fn print_attrs_as_inner(&mut self, attrs: &[hir::Attribute]) {
        self.print_either_attributes(attrs, ast::AttrStyle::Inner)
    }

    fn print_attrs_as_outer(&mut self, attrs: &[hir::Attribute]) {
        self.print_either_attributes(attrs, ast::AttrStyle::Outer)
    }

    fn print_either_attributes(&mut self, attrs: &[hir::Attribute], style: ast::AttrStyle) {
        if attrs.is_empty() {
            return;
        }

        for attr in attrs {
            self.print_attribute_inline(attr, style);
        }
        self.hardbreak_if_not_bol();
    }

    fn print_attribute_inline(&mut self, attr: &hir::Attribute, style: AttrStyle) {
        match &attr {
            hir::Attribute::Unparsed(unparsed) => {
                self.maybe_print_comment(unparsed.span.lo());
                match style {
                    ast::AttrStyle::Inner => self.word("#!["),
                    ast::AttrStyle::Outer => self.word("#["),
                }
                self.print_attr_item(&unparsed, unparsed.span);
                self.word("]");
                self.hardbreak()
            }
            hir::Attribute::Parsed(AttributeKind::DocComment { style, kind, comment, .. }) => {
                self.word(rustc_ast_pretty::pprust::state::doc_comment_to_string(
                    *kind, *style, *comment,
                ));
                self.hardbreak()
            }
            hir::Attribute::Parsed(pa) => {
                self.word("#[attr = ");
                pa.print_attribute(self);
                self.word("]");
                self.hardbreak()
            }
        }
    }

    fn print_attr_item(&mut self, item: &hir::AttrItem, span: Span) {
        let ib = self.ibox(0);
        let path = ast::Path {
            span,
            segments: item
                .path
                .segments
                .iter()
                .map(|i| ast::PathSegment { ident: *i, args: None, id: DUMMY_NODE_ID })
                .collect(),
            tokens: None,
        };

        match &item.args {
            hir::AttrArgs::Delimited(DelimArgs { dspan: _, delim, tokens }) => self
                .print_mac_common(
                    Some(MacHeader::Path(&path)),
                    false,
                    None,
                    *delim,
                    None,
                    &tokens,
                    true,
                    span,
                ),
            hir::AttrArgs::Empty => {
                PrintState::print_path(self, &path, false, 0);
            }
            hir::AttrArgs::Eq { eq_span: _, expr } => {
                PrintState::print_path(self, &path, false, 0);
                self.space();
                self.word_space("=");
                let token_str = self.meta_item_lit_to_string(expr);
                self.word(token_str);
            }
        }
        self.end(ib);
    }

    fn print_node(&mut self, node: Node<'_>) {
        match node {
            Node::Param(a) => self.print_param(a),
            Node::Item(a) => self.print_item(a),
            Node::ForeignItem(a) => self.print_foreign_item(a),
            Node::TraitItem(a) => self.print_trait_item(a),
            Node::ImplItem(a) => self.print_impl_item(a),
            Node::Variant(a) => self.print_variant(a),
            Node::AnonConst(a) => self.print_anon_const(a),
            Node::ConstBlock(a) => self.print_inline_const(a),
            Node::ConstArg(a) => self.print_const_arg(a),
            Node::Expr(a) => self.print_expr(a),
            Node::ExprField(a) => self.print_expr_field(a),
            Node::Stmt(a) => self.print_stmt(a),
            Node::PathSegment(a) => self.print_path_segment(a),
            Node::Ty(a) => self.print_type(a),
            Node::AssocItemConstraint(a) => self.print_assoc_item_constraint(a),
            Node::TraitRef(a) => self.print_trait_ref(a),
            Node::OpaqueTy(_) => panic!("cannot print Node::OpaqueTy"),
            Node::Pat(a) => self.print_pat(a),
            Node::TyPat(a) => self.print_ty_pat(a),
            Node::PatField(a) => self.print_patfield(a),
            Node::PatExpr(a) => self.print_pat_expr(a),
            Node::Arm(a) => self.print_arm(a),
            Node::Infer(_) => self.word("_"),
            Node::PreciseCapturingNonLifetimeArg(param) => self.print_ident(param.ident),
            Node::Block(a) => {
                // Containing cbox, will be closed by print-block at `}`.
                let cb = self.cbox(INDENT_UNIT);
                // Head-ibox, will be closed by print-block after `{`.
                let ib = self.ibox(0);
                self.print_block(a, cb, ib);
            }
            Node::Lifetime(a) => self.print_lifetime(a),
            Node::GenericParam(_) => panic!("cannot print Node::GenericParam"),
            Node::Field(_) => panic!("cannot print Node::Field"),
            // These cases do not carry enough information in the
            // `hir_map` to reconstruct their full structure for pretty
            // printing.
            Node::Ctor(..) => panic!("cannot print isolated Ctor"),
            Node::LetStmt(a) => self.print_local_decl(a),
            Node::Crate(..) => panic!("cannot print Crate"),
            Node::WherePredicate(pred) => self.print_where_predicate(pred),
            Node::Synthetic => unreachable!(),
            Node::Err(_) => self.word("/*ERROR*/"),
        }
    }

    fn print_generic_arg(&mut self, generic_arg: &GenericArg<'_>, elide_lifetimes: bool) {
        match generic_arg {
            GenericArg::Lifetime(lt) if !elide_lifetimes => self.print_lifetime(lt),
            GenericArg::Lifetime(_) => {}
            GenericArg::Type(ty) => self.print_type(ty.as_unambig_ty()),
            GenericArg::Const(ct) => self.print_const_arg(ct.as_unambig_ct()),
            GenericArg::Infer(_inf) => self.word("_"),
        }
    }
}

impl std::ops::Deref for State<'_> {
    type Target = pp::Printer;
    fn deref(&self) -> &Self::Target {
        &self.s
    }
}

impl std::ops::DerefMut for State<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.s
    }
}

impl<'a> PrintState<'a> for State<'a> {
    fn comments(&self) -> Option<&Comments<'a>> {
        self.comments.as_ref()
    }

    fn comments_mut(&mut self) -> Option<&mut Comments<'a>> {
        self.comments.as_mut()
    }

    fn ann_post(&mut self, ident: Ident) {
        self.ann.post(self, AnnNode::Name(&ident.name));
    }

    fn print_generic_args(&mut self, _: &ast::GenericArgs, _colons_before_params: bool) {
        panic!("AST generic args printed by HIR pretty-printer");
    }
}

const INDENT_UNIT: isize = 4;

/// Requires you to pass an input filename and reader so that
/// it can scan the input text for comments to copy forward.
pub fn print_crate<'a>(
    sm: &'a SourceMap,
    krate: &hir::Mod<'_>,
    filename: FileName,
    input: String,
    attrs: &'a dyn Fn(HirId) -> &'a [hir::Attribute],
    ann: &'a dyn PpAnn,
) -> String {
    let mut s = State {
        s: pp::Printer::new(),
        comments: Some(Comments::new(sm, filename, input)),
        attrs,
        ann,
    };

    // When printing the AST, we sometimes need to inject `#[no_std]` here.
    // Since you can't compile the HIR, it's not necessary.

    s.print_mod(krate, (*attrs)(hir::CRATE_HIR_ID));
    s.print_remaining_comments();
    s.s.eof()
}

fn to_string<F>(ann: &dyn PpAnn, f: F) -> String
where
    F: FnOnce(&mut State<'_>),
{
    let mut printer = State { s: pp::Printer::new(), comments: None, attrs: &|_| &[], ann };
    f(&mut printer);
    printer.s.eof()
}

pub fn attribute_to_string(ann: &dyn PpAnn, attr: &hir::Attribute) -> String {
    to_string(ann, |s| s.print_attribute_inline(attr, AttrStyle::Outer))
}

pub fn ty_to_string(ann: &dyn PpAnn, ty: &hir::Ty<'_>) -> String {
    to_string(ann, |s| s.print_type(ty))
}

pub fn qpath_to_string(ann: &dyn PpAnn, segment: &hir::QPath<'_>) -> String {
    to_string(ann, |s| s.print_qpath(segment, false))
}

pub fn pat_to_string(ann: &dyn PpAnn, pat: &hir::Pat<'_>) -> String {
    to_string(ann, |s| s.print_pat(pat))
}

pub fn expr_to_string(ann: &dyn PpAnn, pat: &hir::Expr<'_>) -> String {
    to_string(ann, |s| s.print_expr(pat))
}

pub fn item_to_string(ann: &dyn PpAnn, pat: &hir::Item<'_>) -> String {
    to_string(ann, |s| s.print_item(pat))
}

impl<'a> State<'a> {
    fn bclose_maybe_open(&mut self, span: rustc_span::Span, cb: Option<BoxMarker>) {
        self.maybe_print_comment(span.hi());
        self.break_offset_if_not_bol(1, -INDENT_UNIT);
        self.word("}");
        if let Some(cb) = cb {
            self.end(cb);
        }
    }

    fn bclose(&mut self, span: rustc_span::Span, cb: BoxMarker) {
        self.bclose_maybe_open(span, Some(cb))
    }

    fn commasep_cmnt<T, F, G>(&mut self, b: Breaks, elts: &[T], mut op: F, mut get_span: G)
    where
        F: FnMut(&mut State<'_>, &T),
        G: FnMut(&T) -> rustc_span::Span,
    {
        let rb = self.rbox(0, b);
        let len = elts.len();
        let mut i = 0;
        for elt in elts {
            self.maybe_print_comment(get_span(elt).hi());
            op(self, elt);
            i += 1;
            if i < len {
                self.word(",");
                self.maybe_print_trailing_comment(get_span(elt), Some(get_span(&elts[i]).hi()));
                self.space_if_not_bol();
            }
        }
        self.end(rb);
    }

    fn commasep_exprs(&mut self, b: Breaks, exprs: &[hir::Expr<'_>]) {
        self.commasep_cmnt(b, exprs, |s, e| s.print_expr(e), |e| e.span);
    }

    fn print_mod(&mut self, _mod: &hir::Mod<'_>, attrs: &[hir::Attribute]) {
        self.print_attrs_as_inner(attrs);
        for &item_id in _mod.item_ids {
            self.ann.nested(self, Nested::Item(item_id));
        }
    }

    fn print_opt_lifetime(&mut self, lifetime: &hir::Lifetime) {
        if !lifetime.is_elided() {
            self.print_lifetime(lifetime);
            self.nbsp();
        }
    }

    fn print_type(&mut self, ty: &hir::Ty<'_>) {
        self.maybe_print_comment(ty.span.lo());
        let ib = self.ibox(0);
        match ty.kind {
            hir::TyKind::Slice(ty) => {
                self.word("[");
                self.print_type(ty);
                self.word("]");
            }
            hir::TyKind::Ptr(ref mt) => {
                self.word("*");
                self.print_mt(mt, true);
            }
            hir::TyKind::Ref(lifetime, ref mt) => {
                self.word("&");
                self.print_opt_lifetime(lifetime);
                self.print_mt(mt, false);
            }
            hir::TyKind::Never => {
                self.word("!");
            }
            hir::TyKind::Tup(elts) => {
                self.popen();
                self.commasep(Inconsistent, elts, |s, ty| s.print_type(ty));
                if elts.len() == 1 {
                    self.word(",");
                }
                self.pclose();
            }
            hir::TyKind::BareFn(f) => {
                self.print_ty_fn(f.abi, f.safety, f.decl, None, f.generic_params, f.param_idents);
            }
            hir::TyKind::UnsafeBinder(unsafe_binder) => {
                self.print_unsafe_binder(unsafe_binder);
            }
            hir::TyKind::OpaqueDef(..) => self.word("/*impl Trait*/"),
            hir::TyKind::TraitAscription(bounds) => {
                self.print_bounds("impl", bounds);
            }
            hir::TyKind::Path(ref qpath) => self.print_qpath(qpath, false),
            hir::TyKind::TraitObject(bounds, lifetime) => {
                let syntax = lifetime.tag();
                match syntax {
                    ast::TraitObjectSyntax::Dyn => self.word_nbsp("dyn"),
                    ast::TraitObjectSyntax::DynStar => self.word_nbsp("dyn*"),
                    ast::TraitObjectSyntax::None => {}
                }
                let mut first = true;
                for bound in bounds {
                    if first {
                        first = false;
                    } else {
                        self.nbsp();
                        self.word_space("+");
                    }
                    self.print_poly_trait_ref(bound);
                }
                if !lifetime.is_elided() {
                    self.nbsp();
                    self.word_space("+");
                    self.print_lifetime(lifetime.pointer());
                }
            }
            hir::TyKind::Array(ty, ref length) => {
                self.word("[");
                self.print_type(ty);
                self.word("; ");
                self.print_const_arg(length);
                self.word("]");
            }
            hir::TyKind::Typeof(ref e) => {
                self.word("typeof(");
                self.print_anon_const(e);
                self.word(")");
            }
            hir::TyKind::Err(_) => {
                self.popen();
                self.word("/*ERROR*/");
                self.pclose();
            }
            hir::TyKind::Infer(()) | hir::TyKind::InferDelegation(..) => {
                self.word("_");
            }
            hir::TyKind::Pat(ty, pat) => {
                self.print_type(ty);
                self.word(" is ");
                self.print_ty_pat(pat);
            }
        }
        self.end(ib)
    }

    fn print_unsafe_binder(&mut self, unsafe_binder: &hir::UnsafeBinderTy<'_>) {
        let ib = self.ibox(INDENT_UNIT);
        self.word("unsafe");
        self.print_generic_params(unsafe_binder.generic_params);
        self.nbsp();
        self.print_type(unsafe_binder.inner_ty);
        self.end(ib);
    }

    fn print_foreign_item(&mut self, item: &hir::ForeignItem<'_>) {
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(item.span.lo());
        self.print_attrs_as_outer(self.attrs(item.hir_id()));
        match item.kind {
            hir::ForeignItemKind::Fn(sig, arg_idents, generics) => {
                let (cb, ib) = self.head("");
                self.print_fn(
                    sig.header,
                    Some(item.ident.name),
                    generics,
                    sig.decl,
                    arg_idents,
                    None,
                );
                self.end(ib);
                self.word(";");
                self.end(cb)
            }
            hir::ForeignItemKind::Static(t, m, safety) => {
                self.print_safety(safety);
                let (cb, ib) = self.head("static");
                if m.is_mut() {
                    self.word_space("mut");
                }
                self.print_ident(item.ident);
                self.word_space(":");
                self.print_type(t);
                self.word(";");
                self.end(ib);
                self.end(cb)
            }
            hir::ForeignItemKind::Type => {
                let (cb, ib) = self.head("type");
                self.print_ident(item.ident);
                self.word(";");
                self.end(ib);
                self.end(cb)
            }
        }
    }

    fn print_associated_const(
        &mut self,
        ident: Ident,
        generics: &hir::Generics<'_>,
        ty: &hir::Ty<'_>,
        default: Option<hir::BodyId>,
    ) {
        self.word_space("const");
        self.print_ident(ident);
        self.print_generic_params(generics.params);
        self.word_space(":");
        self.print_type(ty);
        if let Some(expr) = default {
            self.space();
            self.word_space("=");
            self.ann.nested(self, Nested::Body(expr));
        }
        self.print_where_clause(generics);
        self.word(";")
    }

    fn print_associated_type(
        &mut self,
        ident: Ident,
        generics: &hir::Generics<'_>,
        bounds: Option<hir::GenericBounds<'_>>,
        ty: Option<&hir::Ty<'_>>,
    ) {
        self.word_space("type");
        self.print_ident(ident);
        self.print_generic_params(generics.params);
        if let Some(bounds) = bounds {
            self.print_bounds(":", bounds);
        }
        self.print_where_clause(generics);
        if let Some(ty) = ty {
            self.space();
            self.word_space("=");
            self.print_type(ty);
        }
        self.word(";")
    }

    fn print_item(&mut self, item: &hir::Item<'_>) {
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(item.span.lo());
        let attrs = self.attrs(item.hir_id());
        self.print_attrs_as_outer(attrs);
        self.ann.pre(self, AnnNode::Item(item));
        match item.kind {
            hir::ItemKind::ExternCrate(orig_name, ident) => {
                let (cb, ib) = self.head("extern crate");
                if let Some(orig_name) = orig_name {
                    self.print_name(orig_name);
                    self.space();
                    self.word("as");
                    self.space();
                }
                self.print_ident(ident);
                self.word(";");
                self.end(ib);
                self.end(cb);
            }
            hir::ItemKind::Use(path, kind) => {
                let (cb, ib) = self.head("use");
                self.print_path(path, false);

                match kind {
                    hir::UseKind::Single(ident) => {
                        if path.segments.last().unwrap().ident != ident {
                            self.space();
                            self.word_space("as");
                            self.print_ident(ident);
                        }
                        self.word(";");
                    }
                    hir::UseKind::Glob => self.word("::*;"),
                    hir::UseKind::ListStem => self.word("::{};"),
                }
                self.end(ib);
                self.end(cb);
            }
            hir::ItemKind::Static(m, ident, ty, expr) => {
                let (cb, ib) = self.head("static");
                if m.is_mut() {
                    self.word_space("mut");
                }
                self.print_ident(ident);
                self.word_space(":");
                self.print_type(ty);
                self.space();
                self.end(ib);

                self.word_space("=");
                self.ann.nested(self, Nested::Body(expr));
                self.word(";");
                self.end(cb);
            }
            hir::ItemKind::Const(ident, generics, ty, expr) => {
                let (cb, ib) = self.head("const");
                self.print_ident(ident);
                self.print_generic_params(generics.params);
                self.word_space(":");
                self.print_type(ty);
                self.space();
                self.end(ib);

                self.word_space("=");
                self.ann.nested(self, Nested::Body(expr));
                self.print_where_clause(generics);
                self.word(";");
                self.end(cb);
            }
            hir::ItemKind::Fn { ident, sig, generics, body, .. } => {
                let (cb, ib) = self.head("");
                self.print_fn(sig.header, Some(ident.name), generics, sig.decl, &[], Some(body));
                self.word(" ");
                self.end(ib);
                self.end(cb);
                self.ann.nested(self, Nested::Body(body));
            }
            hir::ItemKind::Macro(ident, macro_def, _) => {
                self.print_mac_def(macro_def, &ident, item.span, |_| {});
            }
            hir::ItemKind::Mod(ident, mod_) => {
                let (cb, ib) = self.head("mod");
                self.print_ident(ident);
                self.nbsp();
                self.bopen(ib);
                self.print_mod(mod_, attrs);
                self.bclose(item.span, cb);
            }
            hir::ItemKind::ForeignMod { abi, items } => {
                let (cb, ib) = self.head("extern");
                self.word_nbsp(abi.to_string());
                self.bopen(ib);
                self.print_attrs_as_inner(self.attrs(item.hir_id()));
                for item in items {
                    self.ann.nested(self, Nested::ForeignItem(item.id));
                }
                self.bclose(item.span, cb);
            }
            hir::ItemKind::GlobalAsm { asm, .. } => {
                let (cb, ib) = self.head("global_asm!");
                self.print_inline_asm(asm);
                self.word(";");
                self.end(cb);
                self.end(ib);
            }
            hir::ItemKind::TyAlias(ident, generics, ty) => {
                let (cb, ib) = self.head("type");
                self.print_ident(ident);
                self.print_generic_params(generics.params);
                self.end(ib);

                self.print_where_clause(generics);
                self.space();
                self.word_space("=");
                self.print_type(ty);
                self.word(";");
                self.end(cb);
            }
            hir::ItemKind::Enum(ident, generics, ref enum_def) => {
                self.print_enum_def(ident.name, generics, enum_def, item.span);
            }
            hir::ItemKind::Struct(ident, generics, ref struct_def) => {
                let (cb, ib) = self.head("struct");
                self.print_struct(ident.name, generics, struct_def, item.span, true, cb, ib);
            }
            hir::ItemKind::Union(ident, generics, ref struct_def) => {
                let (cb, ib) = self.head("union");
                self.print_struct(ident.name, generics, struct_def, item.span, true, cb, ib);
            }
            hir::ItemKind::Impl(&hir::Impl {
                constness,
                safety,
                polarity,
                defaultness,
                defaultness_span: _,
                generics,
                ref of_trait,
                self_ty,
                items,
            }) => {
                let (cb, ib) = self.head("");
                self.print_defaultness(defaultness);
                self.print_safety(safety);
                self.word_nbsp("impl");

                if let hir::Constness::Const = constness {
                    self.word_nbsp("const");
                }

                if !generics.params.is_empty() {
                    self.print_generic_params(generics.params);
                    self.space();
                }

                if let hir::ImplPolarity::Negative(_) = polarity {
                    self.word("!");
                }

                if let Some(t) = of_trait {
                    self.print_trait_ref(t);
                    self.space();
                    self.word_space("for");
                }

                self.print_type(self_ty);
                self.print_where_clause(generics);

                self.space();
                self.bopen(ib);
                self.print_attrs_as_inner(attrs);
                for impl_item in items {
                    self.ann.nested(self, Nested::ImplItem(impl_item.id));
                }
                self.bclose(item.span, cb);
            }
            hir::ItemKind::Trait(is_auto, safety, ident, generics, bounds, trait_items) => {
                let (cb, ib) = self.head("");
                self.print_is_auto(is_auto);
                self.print_safety(safety);
                self.word_nbsp("trait");
                self.print_ident(ident);
                self.print_generic_params(generics.params);
                self.print_bounds(":", bounds);
                self.print_where_clause(generics);
                self.word(" ");
                self.bopen(ib);
                for trait_item in trait_items {
                    self.ann.nested(self, Nested::TraitItem(trait_item.id));
                }
                self.bclose(item.span, cb);
            }
            hir::ItemKind::TraitAlias(ident, generics, bounds) => {
                let (cb, ib) = self.head("trait");
                self.print_ident(ident);
                self.print_generic_params(generics.params);
                self.nbsp();
                self.print_bounds("=", bounds);
                self.print_where_clause(generics);
                self.word(";");
                self.end(ib);
                self.end(cb);
            }
        }
        self.ann.post(self, AnnNode::Item(item))
    }

    fn print_trait_ref(&mut self, t: &hir::TraitRef<'_>) {
        self.print_path(t.path, false);
    }

    fn print_formal_generic_params(&mut self, generic_params: &[hir::GenericParam<'_>]) {
        if !generic_params.is_empty() {
            self.word("for");
            self.print_generic_params(generic_params);
            self.nbsp();
        }
    }

    fn print_poly_trait_ref(&mut self, t: &hir::PolyTraitRef<'_>) {
        let hir::TraitBoundModifiers { constness, polarity } = t.modifiers;
        match constness {
            hir::BoundConstness::Never => {}
            hir::BoundConstness::Always(_) => self.word("const"),
            hir::BoundConstness::Maybe(_) => self.word("~const"),
        }
        match polarity {
            hir::BoundPolarity::Positive => {}
            hir::BoundPolarity::Negative(_) => self.word("!"),
            hir::BoundPolarity::Maybe(_) => self.word("?"),
        }
        self.print_formal_generic_params(t.bound_generic_params);
        self.print_trait_ref(&t.trait_ref);
    }

    fn print_enum_def(
        &mut self,
        name: Symbol,
        generics: &hir::Generics<'_>,
        enum_def: &hir::EnumDef<'_>,
        span: rustc_span::Span,
    ) {
        let (cb, ib) = self.head("enum");
        self.print_name(name);
        self.print_generic_params(generics.params);
        self.print_where_clause(generics);
        self.space();
        self.print_variants(enum_def.variants, span, cb, ib);
    }

    fn print_variants(
        &mut self,
        variants: &[hir::Variant<'_>],
        span: rustc_span::Span,
        cb: BoxMarker,
        ib: BoxMarker,
    ) {
        self.bopen(ib);
        for v in variants {
            self.space_if_not_bol();
            self.maybe_print_comment(v.span.lo());
            self.print_attrs_as_outer(self.attrs(v.hir_id));
            let ib = self.ibox(INDENT_UNIT);
            self.print_variant(v);
            self.word(",");
            self.end(ib);
            self.maybe_print_trailing_comment(v.span, None);
        }
        self.bclose(span, cb)
    }

    fn print_defaultness(&mut self, defaultness: hir::Defaultness) {
        match defaultness {
            hir::Defaultness::Default { .. } => self.word_nbsp("default"),
            hir::Defaultness::Final => (),
        }
    }

    fn print_struct(
        &mut self,
        name: Symbol,
        generics: &hir::Generics<'_>,
        struct_def: &hir::VariantData<'_>,
        span: rustc_span::Span,
        print_finalizer: bool,
        cb: BoxMarker,
        ib: BoxMarker,
    ) {
        self.print_name(name);
        self.print_generic_params(generics.params);
        match struct_def {
            hir::VariantData::Tuple(..) | hir::VariantData::Unit(..) => {
                if let hir::VariantData::Tuple(..) = struct_def {
                    self.popen();
                    self.commasep(Inconsistent, struct_def.fields(), |s, field| {
                        s.maybe_print_comment(field.span.lo());
                        s.print_attrs_as_outer(s.attrs(field.hir_id));
                        s.print_type(field.ty);
                    });
                    self.pclose();
                }
                self.print_where_clause(generics);
                if print_finalizer {
                    self.word(";");
                }
                self.end(ib);
                self.end(cb);
            }
            hir::VariantData::Struct { .. } => {
                self.print_where_clause(generics);
                self.nbsp();
                self.bopen(ib);
                self.hardbreak_if_not_bol();

                for field in struct_def.fields() {
                    self.hardbreak_if_not_bol();
                    self.maybe_print_comment(field.span.lo());
                    self.print_attrs_as_outer(self.attrs(field.hir_id));
                    self.print_ident(field.ident);
                    self.word_nbsp(":");
                    self.print_type(field.ty);
                    self.word(",");
                }

                self.bclose(span, cb)
            }
        }
    }

    pub fn print_variant(&mut self, v: &hir::Variant<'_>) {
        let (cb, ib) = self.head("");
        let generics = hir::Generics::empty();
        self.print_struct(v.ident.name, generics, &v.data, v.span, false, cb, ib);
        if let Some(ref d) = v.disr_expr {
            self.space();
            self.word_space("=");
            self.print_anon_const(d);
        }
    }

    fn print_method_sig(
        &mut self,
        ident: Ident,
        m: &hir::FnSig<'_>,
        generics: &hir::Generics<'_>,
        arg_idents: &[Option<Ident>],
        body_id: Option<hir::BodyId>,
    ) {
        self.print_fn(m.header, Some(ident.name), generics, m.decl, arg_idents, body_id);
    }

    fn print_trait_item(&mut self, ti: &hir::TraitItem<'_>) {
        self.ann.pre(self, AnnNode::SubItem(ti.hir_id()));
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(ti.span.lo());
        self.print_attrs_as_outer(self.attrs(ti.hir_id()));
        match ti.kind {
            hir::TraitItemKind::Const(ty, default) => {
                self.print_associated_const(ti.ident, ti.generics, ty, default);
            }
            hir::TraitItemKind::Fn(ref sig, hir::TraitFn::Required(arg_idents)) => {
                self.print_method_sig(ti.ident, sig, ti.generics, arg_idents, None);
                self.word(";");
            }
            hir::TraitItemKind::Fn(ref sig, hir::TraitFn::Provided(body)) => {
                let (cb, ib) = self.head("");
                self.print_method_sig(ti.ident, sig, ti.generics, &[], Some(body));
                self.nbsp();
                self.end(ib);
                self.end(cb);
                self.ann.nested(self, Nested::Body(body));
            }
            hir::TraitItemKind::Type(bounds, default) => {
                self.print_associated_type(ti.ident, ti.generics, Some(bounds), default);
            }
        }
        self.ann.post(self, AnnNode::SubItem(ti.hir_id()))
    }

    fn print_impl_item(&mut self, ii: &hir::ImplItem<'_>) {
        self.ann.pre(self, AnnNode::SubItem(ii.hir_id()));
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(ii.span.lo());
        self.print_attrs_as_outer(self.attrs(ii.hir_id()));

        match ii.kind {
            hir::ImplItemKind::Const(ty, expr) => {
                self.print_associated_const(ii.ident, ii.generics, ty, Some(expr));
            }
            hir::ImplItemKind::Fn(ref sig, body) => {
                let (cb, ib) = self.head("");
                self.print_method_sig(ii.ident, sig, ii.generics, &[], Some(body));
                self.nbsp();
                self.end(ib);
                self.end(cb);
                self.ann.nested(self, Nested::Body(body));
            }
            hir::ImplItemKind::Type(ty) => {
                self.print_associated_type(ii.ident, ii.generics, None, Some(ty));
            }
        }
        self.ann.post(self, AnnNode::SubItem(ii.hir_id()))
    }

    fn print_local(
        &mut self,
        super_: bool,
        init: Option<&hir::Expr<'_>>,
        els: Option<&hir::Block<'_>>,
        decl: impl Fn(&mut Self),
    ) {
        self.space_if_not_bol();
        let ibm1 = self.ibox(INDENT_UNIT);
        if super_ {
            self.word_nbsp("super");
        }
        self.word_nbsp("let");

        let ibm2 = self.ibox(INDENT_UNIT);
        decl(self);
        self.end(ibm2);

        if let Some(init) = init {
            self.nbsp();
            self.word_space("=");
            self.print_expr(init);
        }

        if let Some(els) = els {
            self.nbsp();
            self.word_space("else");
            // containing cbox, will be closed by print-block at `}`
            let cb = self.cbox(0);
            // head-box, will be closed by print-block after `{`
            let ib = self.ibox(0);
            self.print_block(els, cb, ib);
        }

        self.end(ibm1)
    }

    fn print_stmt(&mut self, st: &hir::Stmt<'_>) {
        self.maybe_print_comment(st.span.lo());
        match st.kind {
            hir::StmtKind::Let(loc) => {
                self.print_local(loc.super_.is_some(), loc.init, loc.els, |this| {
                    this.print_local_decl(loc)
                });
            }
            hir::StmtKind::Item(item) => self.ann.nested(self, Nested::Item(item)),
            hir::StmtKind::Expr(expr) => {
                self.space_if_not_bol();
                self.print_expr(expr);
            }
            hir::StmtKind::Semi(expr) => {
                self.space_if_not_bol();
                self.print_expr(expr);
                self.word(";");
            }
        }
        if stmt_ends_with_semi(&st.kind) {
            self.word(";");
        }
        self.maybe_print_trailing_comment(st.span, None)
    }

    fn print_block(&mut self, blk: &hir::Block<'_>, cb: BoxMarker, ib: BoxMarker) {
        self.print_block_with_attrs(blk, &[], cb, ib)
    }

    fn print_block_unclosed(&mut self, blk: &hir::Block<'_>, ib: BoxMarker) {
        self.print_block_maybe_unclosed(blk, &[], None, ib)
    }

    fn print_block_with_attrs(
        &mut self,
        blk: &hir::Block<'_>,
        attrs: &[hir::Attribute],
        cb: BoxMarker,
        ib: BoxMarker,
    ) {
        self.print_block_maybe_unclosed(blk, attrs, Some(cb), ib)
    }

    fn print_block_maybe_unclosed(
        &mut self,
        blk: &hir::Block<'_>,
        attrs: &[hir::Attribute],
        cb: Option<BoxMarker>,
        ib: BoxMarker,
    ) {
        match blk.rules {
            hir::BlockCheckMode::UnsafeBlock(..) => self.word_space("unsafe"),
            hir::BlockCheckMode::DefaultBlock => (),
        }
        self.maybe_print_comment(blk.span.lo());
        self.ann.pre(self, AnnNode::Block(blk));
        self.bopen(ib);

        self.print_attrs_as_inner(attrs);

        for st in blk.stmts {
            self.print_stmt(st);
        }
        if let Some(expr) = blk.expr {
            self.space_if_not_bol();
            self.print_expr(expr);
            self.maybe_print_trailing_comment(expr.span, Some(blk.span.hi()));
        }
        self.bclose_maybe_open(blk.span, cb);
        self.ann.post(self, AnnNode::Block(blk))
    }

    fn print_else(&mut self, els: Option<&hir::Expr<'_>>) {
        if let Some(els_inner) = els {
            match els_inner.kind {
                // Another `else if` block.
                hir::ExprKind::If(i, hir::Expr { kind: hir::ExprKind::Block(t, None), .. }, e) => {
                    let cb = self.cbox(0);
                    let ib = self.ibox(0);
                    self.word(" else if ");
                    self.print_expr_as_cond(i);
                    self.space();
                    self.print_block(t, cb, ib);
                    self.print_else(e);
                }
                // Final `else` block.
                hir::ExprKind::Block(b, None) => {
                    let cb = self.cbox(0);
                    let ib = self.ibox(0);
                    self.word(" else ");
                    self.print_block(b, cb, ib);
                }
                // Constraints would be great here!
                _ => {
                    panic!("print_if saw if with weird alternative");
                }
            }
        }
    }

    fn print_if(
        &mut self,
        test: &hir::Expr<'_>,
        blk: &hir::Expr<'_>,
        elseopt: Option<&hir::Expr<'_>>,
    ) {
        match blk.kind {
            hir::ExprKind::Block(blk, None) => {
                let cb = self.cbox(0);
                let ib = self.ibox(0);
                self.word_nbsp("if");
                self.print_expr_as_cond(test);
                self.space();
                self.print_block(blk, cb, ib);
                self.print_else(elseopt)
            }
            _ => panic!("non-block then expr"),
        }
    }

    fn print_anon_const(&mut self, constant: &hir::AnonConst) {
        self.ann.nested(self, Nested::Body(constant.body))
    }

    fn print_const_arg(&mut self, const_arg: &hir::ConstArg<'_>) {
        match &const_arg.kind {
            ConstArgKind::Path(qpath) => self.print_qpath(qpath, true),
            ConstArgKind::Anon(anon) => self.print_anon_const(anon),
            ConstArgKind::Infer(..) => self.word("_"),
        }
    }

    fn print_call_post(&mut self, args: &[hir::Expr<'_>]) {
        self.popen();
        self.commasep_exprs(Inconsistent, args);
        self.pclose()
    }

    /// Prints an expr using syntax that's acceptable in a condition position, such as the `cond` in
    /// `if cond { ... }`.
    fn print_expr_as_cond(&mut self, expr: &hir::Expr<'_>) {
        self.print_expr_cond_paren(expr, Self::cond_needs_par(expr))
    }

    /// Prints `expr` or `(expr)` when `needs_par` holds.
    fn print_expr_cond_paren(&mut self, expr: &hir::Expr<'_>, needs_par: bool) {
        if needs_par {
            self.popen();
        }
        if let hir::ExprKind::DropTemps(actual_expr) = expr.kind {
            self.print_expr(actual_expr);
        } else {
            self.print_expr(expr);
        }
        if needs_par {
            self.pclose();
        }
    }

    /// Print a `let pat = expr` expression.
    fn print_let(&mut self, pat: &hir::Pat<'_>, ty: Option<&hir::Ty<'_>>, init: &hir::Expr<'_>) {
        self.word_space("let");
        self.print_pat(pat);
        if let Some(ty) = ty {
            self.word_space(":");
            self.print_type(ty);
        }
        self.space();
        self.word_space("=");
        let npals = || parser::needs_par_as_let_scrutinee(self.precedence(init));
        self.print_expr_cond_paren(init, Self::cond_needs_par(init) || npals())
    }

    // Does `expr` need parentheses when printed in a condition position?
    //
    // These cases need parens due to the parse error observed in #26461: `if return {}`
    // parses as the erroneous construct `if (return {})`, not `if (return) {}`.
    fn cond_needs_par(expr: &hir::Expr<'_>) -> bool {
        match expr.kind {
            hir::ExprKind::Break(..) | hir::ExprKind::Closure { .. } | hir::ExprKind::Ret(..) => {
                true
            }
            _ => contains_exterior_struct_lit(expr),
        }
    }

    fn print_expr_vec(&mut self, exprs: &[hir::Expr<'_>]) {
        let ib = self.ibox(INDENT_UNIT);
        self.word("[");
        self.commasep_exprs(Inconsistent, exprs);
        self.word("]");
        self.end(ib)
    }

    fn print_inline_const(&mut self, constant: &hir::ConstBlock) {
        let ib = self.ibox(INDENT_UNIT);
        self.word_space("const");
        self.ann.nested(self, Nested::Body(constant.body));
        self.end(ib)
    }

    fn print_expr_repeat(&mut self, element: &hir::Expr<'_>, count: &hir::ConstArg<'_>) {
        let ib = self.ibox(INDENT_UNIT);
        self.word("[");
        self.print_expr(element);
        self.word_space(";");
        self.print_const_arg(count);
        self.word("]");
        self.end(ib)
    }

    fn print_expr_struct(
        &mut self,
        qpath: &hir::QPath<'_>,
        fields: &[hir::ExprField<'_>],
        wth: hir::StructTailExpr<'_>,
    ) {
        self.print_qpath(qpath, true);
        self.nbsp();
        self.word_space("{");
        self.commasep_cmnt(Consistent, fields, |s, field| s.print_expr_field(field), |f| f.span);
        match wth {
            hir::StructTailExpr::Base(expr) => {
                let ib = self.ibox(INDENT_UNIT);
                if !fields.is_empty() {
                    self.word(",");
                    self.space();
                }
                self.word("..");
                self.print_expr(expr);
                self.end(ib);
            }
            hir::StructTailExpr::DefaultFields(_) => {
                let ib = self.ibox(INDENT_UNIT);
                if !fields.is_empty() {
                    self.word(",");
                    self.space();
                }
                self.word("..");
                self.end(ib);
            }
            hir::StructTailExpr::None => {}
        }
        self.space();
        self.word("}");
    }

    fn print_expr_field(&mut self, field: &hir::ExprField<'_>) {
        let cb = self.cbox(INDENT_UNIT);
        self.print_attrs_as_outer(self.attrs(field.hir_id));
        if !field.is_shorthand {
            self.print_ident(field.ident);
            self.word_space(":");
        }
        self.print_expr(field.expr);
        self.end(cb)
    }

    fn print_expr_tup(&mut self, exprs: &[hir::Expr<'_>]) {
        self.popen();
        self.commasep_exprs(Inconsistent, exprs);
        if exprs.len() == 1 {
            self.word(",");
        }
        self.pclose()
    }

    fn print_expr_call(&mut self, func: &hir::Expr<'_>, args: &[hir::Expr<'_>]) {
        let needs_paren = match func.kind {
            hir::ExprKind::Field(..) => true,
            _ => self.precedence(func) < ExprPrecedence::Unambiguous,
        };

        self.print_expr_cond_paren(func, needs_paren);
        self.print_call_post(args)
    }

    fn print_expr_method_call(
        &mut self,
        segment: &hir::PathSegment<'_>,
        receiver: &hir::Expr<'_>,
        args: &[hir::Expr<'_>],
    ) {
        let base_args = args;
        self.print_expr_cond_paren(
            receiver,
            self.precedence(receiver) < ExprPrecedence::Unambiguous,
        );
        self.word(".");
        self.print_ident(segment.ident);

        let generic_args = segment.args();
        if !generic_args.args.is_empty() || !generic_args.constraints.is_empty() {
            self.print_generic_args(generic_args, true);
        }

        self.print_call_post(base_args)
    }

    fn print_expr_binary(&mut self, op: hir::BinOpKind, lhs: &hir::Expr<'_>, rhs: &hir::Expr<'_>) {
        let binop_prec = op.precedence();
        let left_prec = self.precedence(lhs);
        let right_prec = self.precedence(rhs);

        let (mut left_needs_paren, right_needs_paren) = match op.fixity() {
            Fixity::Left => (left_prec < binop_prec, right_prec <= binop_prec),
            Fixity::Right => (left_prec <= binop_prec, right_prec < binop_prec),
            Fixity::None => (left_prec <= binop_prec, right_prec <= binop_prec),
        };

        match (&lhs.kind, op) {
            // These cases need parens: `x as i32 < y` has the parser thinking that `i32 < y` is
            // the beginning of a path type. It starts trying to parse `x as (i32 < y ...` instead
            // of `(x as i32) < ...`. We need to convince it _not_ to do that.
            (&hir::ExprKind::Cast { .. }, hir::BinOpKind::Lt | hir::BinOpKind::Shl) => {
                left_needs_paren = true;
            }
            (&hir::ExprKind::Let { .. }, _) if !parser::needs_par_as_let_scrutinee(binop_prec) => {
                left_needs_paren = true;
            }
            _ => {}
        }

        self.print_expr_cond_paren(lhs, left_needs_paren);
        self.space();
        self.word_space(op.as_str());
        self.print_expr_cond_paren(rhs, right_needs_paren);
    }

    fn print_expr_unary(&mut self, op: hir::UnOp, expr: &hir::Expr<'_>) {
        self.word(op.as_str());
        self.print_expr_cond_paren(expr, self.precedence(expr) < ExprPrecedence::Prefix);
    }

    fn print_expr_addr_of(
        &mut self,
        kind: hir::BorrowKind,
        mutability: hir::Mutability,
        expr: &hir::Expr<'_>,
    ) {
        self.word("&");
        match kind {
            hir::BorrowKind::Ref => self.print_mutability(mutability, false),
            hir::BorrowKind::Raw => {
                self.word_nbsp("raw");
                self.print_mutability(mutability, true);
            }
        }
        self.print_expr_cond_paren(expr, self.precedence(expr) < ExprPrecedence::Prefix);
    }

    fn print_literal(&mut self, lit: &hir::Lit) {
        self.maybe_print_comment(lit.span.lo());
        self.word(lit.node.to_string())
    }

    fn print_inline_asm(&mut self, asm: &hir::InlineAsm<'_>) {
        enum AsmArg<'a> {
            Template(String),
            Operand(&'a hir::InlineAsmOperand<'a>),
            Options(ast::InlineAsmOptions),
        }

        let mut args = vec![AsmArg::Template(ast::InlineAsmTemplatePiece::to_string(asm.template))];
        args.extend(asm.operands.iter().map(|(o, _)| AsmArg::Operand(o)));
        if !asm.options.is_empty() {
            args.push(AsmArg::Options(asm.options));
        }

        self.popen();
        self.commasep(Consistent, &args, |s, arg| match *arg {
            AsmArg::Template(ref template) => s.print_string(template, ast::StrStyle::Cooked),
            AsmArg::Operand(op) => match *op {
                hir::InlineAsmOperand::In { reg, expr } => {
                    s.word("in");
                    s.popen();
                    s.word(format!("{reg}"));
                    s.pclose();
                    s.space();
                    s.print_expr(expr);
                }
                hir::InlineAsmOperand::Out { reg, late, ref expr } => {
                    s.word(if late { "lateout" } else { "out" });
                    s.popen();
                    s.word(format!("{reg}"));
                    s.pclose();
                    s.space();
                    match expr {
                        Some(expr) => s.print_expr(expr),
                        None => s.word("_"),
                    }
                }
                hir::InlineAsmOperand::InOut { reg, late, expr } => {
                    s.word(if late { "inlateout" } else { "inout" });
                    s.popen();
                    s.word(format!("{reg}"));
                    s.pclose();
                    s.space();
                    s.print_expr(expr);
                }
                hir::InlineAsmOperand::SplitInOut { reg, late, in_expr, ref out_expr } => {
                    s.word(if late { "inlateout" } else { "inout" });
                    s.popen();
                    s.word(format!("{reg}"));
                    s.pclose();
                    s.space();
                    s.print_expr(in_expr);
                    s.space();
                    s.word_space("=>");
                    match out_expr {
                        Some(out_expr) => s.print_expr(out_expr),
                        None => s.word("_"),
                    }
                }
                hir::InlineAsmOperand::Const { ref anon_const } => {
                    s.word("const");
                    s.space();
                    // Not using `print_inline_const` to avoid additional `const { ... }`
                    s.ann.nested(s, Nested::Body(anon_const.body))
                }
                hir::InlineAsmOperand::SymFn { ref expr } => {
                    s.word("sym_fn");
                    s.space();
                    s.print_expr(expr);
                }
                hir::InlineAsmOperand::SymStatic { ref path, def_id: _ } => {
                    s.word("sym_static");
                    s.space();
                    s.print_qpath(path, true);
                }
                hir::InlineAsmOperand::Label { block } => {
                    let (cb, ib) = s.head("label");
                    s.print_block(block, cb, ib);
                }
            },
            AsmArg::Options(opts) => {
                s.word("options");
                s.popen();
                s.commasep(Inconsistent, &opts.human_readable_names(), |s, &opt| {
                    s.word(opt);
                });
                s.pclose();
            }
        });
        self.pclose();
    }

    fn print_expr(&mut self, expr: &hir::Expr<'_>) {
        self.maybe_print_comment(expr.span.lo());
        self.print_attrs_as_outer(self.attrs(expr.hir_id));
        let ib = self.ibox(INDENT_UNIT);
        self.ann.pre(self, AnnNode::Expr(expr));
        match expr.kind {
            hir::ExprKind::Array(exprs) => {
                self.print_expr_vec(exprs);
            }
            hir::ExprKind::ConstBlock(ref anon_const) => {
                self.print_inline_const(anon_const);
            }
            hir::ExprKind::Repeat(element, ref count) => {
                self.print_expr_repeat(element, count);
            }
            hir::ExprKind::Struct(qpath, fields, wth) => {
                self.print_expr_struct(qpath, fields, wth);
            }
            hir::ExprKind::Tup(exprs) => {
                self.print_expr_tup(exprs);
            }
            hir::ExprKind::Call(func, args) => {
                self.print_expr_call(func, args);
            }
            hir::ExprKind::MethodCall(segment, receiver, args, _) => {
                self.print_expr_method_call(segment, receiver, args);
            }
            hir::ExprKind::Use(expr, _) => {
                self.print_expr(expr);
                self.word(".use");
            }
            hir::ExprKind::Binary(op, lhs, rhs) => {
                self.print_expr_binary(op.node, lhs, rhs);
            }
            hir::ExprKind::Unary(op, expr) => {
                self.print_expr_unary(op, expr);
            }
            hir::ExprKind::AddrOf(k, m, expr) => {
                self.print_expr_addr_of(k, m, expr);
            }
            hir::ExprKind::Lit(lit) => {
                self.print_literal(lit);
            }
            hir::ExprKind::Cast(expr, ty) => {
                self.print_expr_cond_paren(expr, self.precedence(expr) < ExprPrecedence::Cast);
                self.space();
                self.word_space("as");
                self.print_type(ty);
            }
            hir::ExprKind::Type(expr, ty) => {
                self.word("type_ascribe!(");
                let ib = self.ibox(0);
                self.print_expr(expr);

                self.word(",");
                self.space_if_not_bol();
                self.print_type(ty);

                self.end(ib);
                self.word(")");
            }
            hir::ExprKind::DropTemps(init) => {
                // Print `{`:
                let cb = self.cbox(0);
                let ib = self.ibox(0);
                self.bopen(ib);

                // Print `let _t = $init;`:
                let temp = Ident::with_dummy_span(sym::_t);
                self.print_local(false, Some(init), None, |this| this.print_ident(temp));
                self.word(";");

                // Print `_t`:
                self.space_if_not_bol();
                self.print_ident(temp);

                // Print `}`:
                self.bclose_maybe_open(expr.span, Some(cb));
            }
            hir::ExprKind::Let(&hir::LetExpr { pat, ty, init, .. }) => {
                self.print_let(pat, ty, init);
            }
            hir::ExprKind::If(test, blk, elseopt) => {
                self.print_if(test, blk, elseopt);
            }
            hir::ExprKind::Loop(blk, opt_label, _, _) => {
                let cb = self.cbox(0);
                let ib = self.ibox(0);
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }
                self.word_nbsp("loop");
                self.print_block(blk, cb, ib);
            }
            hir::ExprKind::Match(expr, arms, _) => {
                let cb = self.cbox(0);
                let ib = self.ibox(0);
                self.word_nbsp("match");
                self.print_expr_as_cond(expr);
                self.space();
                self.bopen(ib);
                for arm in arms {
                    self.print_arm(arm);
                }
                self.bclose(expr.span, cb);
            }
            hir::ExprKind::Closure(&hir::Closure {
                binder,
                constness,
                capture_clause,
                bound_generic_params,
                fn_decl,
                body,
                fn_decl_span: _,
                fn_arg_span: _,
                kind: _,
                def_id: _,
            }) => {
                self.print_closure_binder(binder, bound_generic_params);
                self.print_constness(constness);
                self.print_capture_clause(capture_clause);

                self.print_closure_params(fn_decl, body);
                self.space();

                // This is a bare expression.
                self.ann.nested(self, Nested::Body(body));
            }
            hir::ExprKind::Block(blk, opt_label) => {
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }
                // containing cbox, will be closed by print-block at `}`
                let cb = self.cbox(0);
                // head-box, will be closed by print-block after `{`
                let ib = self.ibox(0);
                self.print_block(blk, cb, ib);
            }
            hir::ExprKind::Assign(lhs, rhs, _) => {
                self.print_expr_cond_paren(lhs, self.precedence(lhs) <= ExprPrecedence::Assign);
                self.space();
                self.word_space("=");
                self.print_expr_cond_paren(rhs, self.precedence(rhs) < ExprPrecedence::Assign);
            }
            hir::ExprKind::AssignOp(op, lhs, rhs) => {
                self.print_expr_cond_paren(lhs, self.precedence(lhs) <= ExprPrecedence::Assign);
                self.space();
                self.word_space(op.node.as_str());
                self.print_expr_cond_paren(rhs, self.precedence(rhs) < ExprPrecedence::Assign);
            }
            hir::ExprKind::Field(expr, ident) => {
                self.print_expr_cond_paren(
                    expr,
                    self.precedence(expr) < ExprPrecedence::Unambiguous,
                );
                self.word(".");
                self.print_ident(ident);
            }
            hir::ExprKind::Index(expr, index, _) => {
                self.print_expr_cond_paren(
                    expr,
                    self.precedence(expr) < ExprPrecedence::Unambiguous,
                );
                self.word("[");
                self.print_expr(index);
                self.word("]");
            }
            hir::ExprKind::Path(ref qpath) => self.print_qpath(qpath, true),
            hir::ExprKind::Break(destination, opt_expr) => {
                self.word("break");
                if let Some(label) = destination.label {
                    self.space();
                    self.print_ident(label.ident);
                }
                if let Some(expr) = opt_expr {
                    self.space();
                    self.print_expr_cond_paren(expr, self.precedence(expr) < ExprPrecedence::Jump);
                }
            }
            hir::ExprKind::Continue(destination) => {
                self.word("continue");
                if let Some(label) = destination.label {
                    self.space();
                    self.print_ident(label.ident);
                }
            }
            hir::ExprKind::Ret(result) => {
                self.word("return");
                if let Some(expr) = result {
                    self.word(" ");
                    self.print_expr_cond_paren(expr, self.precedence(expr) < ExprPrecedence::Jump);
                }
            }
            hir::ExprKind::Become(result) => {
                self.word("become");
                self.word(" ");
                self.print_expr_cond_paren(result, self.precedence(result) < ExprPrecedence::Jump);
            }
            hir::ExprKind::InlineAsm(asm) => {
                self.word("asm!");
                self.print_inline_asm(asm);
            }
            hir::ExprKind::OffsetOf(container, fields) => {
                self.word("offset_of!(");
                self.print_type(container);
                self.word(",");
                self.space();

                if let Some((&first, rest)) = fields.split_first() {
                    self.print_ident(first);

                    for &field in rest {
                        self.word(".");
                        self.print_ident(field);
                    }
                }

                self.word(")");
            }
            hir::ExprKind::UnsafeBinderCast(kind, expr, ty) => {
                match kind {
                    ast::UnsafeBinderCastKind::Wrap => self.word("wrap_binder!("),
                    ast::UnsafeBinderCastKind::Unwrap => self.word("unwrap_binder!("),
                }
                self.print_expr(expr);
                if let Some(ty) = ty {
                    self.word(",");
                    self.space();
                    self.print_type(ty);
                }
                self.word(")");
            }
            hir::ExprKind::Yield(expr, _) => {
                self.word_space("yield");
                self.print_expr_cond_paren(expr, self.precedence(expr) < ExprPrecedence::Jump);
            }
            hir::ExprKind::Err(_) => {
                self.popen();
                self.word("/*ERROR*/");
                self.pclose();
            }
        }
        self.ann.post(self, AnnNode::Expr(expr));
        self.end(ib)
    }

    fn print_local_decl(&mut self, loc: &hir::LetStmt<'_>) {
        self.print_pat(loc.pat);
        if let Some(ty) = loc.ty {
            self.word_space(":");
            self.print_type(ty);
        }
    }

    fn print_name(&mut self, name: Symbol) {
        self.print_ident(Ident::with_dummy_span(name))
    }

    fn print_path<R>(&mut self, path: &hir::Path<'_, R>, colons_before_params: bool) {
        self.maybe_print_comment(path.span.lo());

        for (i, segment) in path.segments.iter().enumerate() {
            if i > 0 {
                self.word("::")
            }
            if segment.ident.name != kw::PathRoot {
                self.print_ident(segment.ident);
                self.print_generic_args(segment.args(), colons_before_params);
            }
        }
    }

    fn print_path_segment(&mut self, segment: &hir::PathSegment<'_>) {
        if segment.ident.name != kw::PathRoot {
            self.print_ident(segment.ident);
            self.print_generic_args(segment.args(), false);
        }
    }

    fn print_qpath(&mut self, qpath: &hir::QPath<'_>, colons_before_params: bool) {
        match *qpath {
            hir::QPath::Resolved(None, path) => self.print_path(path, colons_before_params),
            hir::QPath::Resolved(Some(qself), path) => {
                self.word("<");
                self.print_type(qself);
                self.space();
                self.word_space("as");

                for (i, segment) in path.segments[..path.segments.len() - 1].iter().enumerate() {
                    if i > 0 {
                        self.word("::")
                    }
                    if segment.ident.name != kw::PathRoot {
                        self.print_ident(segment.ident);
                        self.print_generic_args(segment.args(), colons_before_params);
                    }
                }

                self.word(">");
                self.word("::");
                let item_segment = path.segments.last().unwrap();
                self.print_ident(item_segment.ident);
                self.print_generic_args(item_segment.args(), colons_before_params)
            }
            hir::QPath::TypeRelative(qself, item_segment) => {
                // If we've got a compound-qualified-path, let's push an additional pair of angle
                // brackets, so that we pretty-print `<<A::B>::C>` as `<A::B>::C`, instead of just
                // `A::B::C` (since the latter could be ambiguous to the user)
                if let hir::TyKind::Path(hir::QPath::Resolved(None, _)) = qself.kind {
                    self.print_type(qself);
                } else {
                    self.word("<");
                    self.print_type(qself);
                    self.word(">");
                }

                self.word("::");
                self.print_ident(item_segment.ident);
                self.print_generic_args(item_segment.args(), colons_before_params)
            }
            hir::QPath::LangItem(lang_item, span) => {
                self.word("#[lang = \"");
                self.print_ident(Ident::new(lang_item.name(), span));
                self.word("\"]");
            }
        }
    }

    fn print_generic_args(
        &mut self,
        generic_args: &hir::GenericArgs<'_>,
        colons_before_params: bool,
    ) {
        match generic_args.parenthesized {
            hir::GenericArgsParentheses::No => {
                let start = if colons_before_params { "::<" } else { "<" };
                let empty = Cell::new(true);
                let start_or_comma = |this: &mut Self| {
                    if empty.get() {
                        empty.set(false);
                        this.word(start)
                    } else {
                        this.word_space(",")
                    }
                };

                let mut nonelided_generic_args: bool = false;
                let elide_lifetimes = generic_args.args.iter().all(|arg| match arg {
                    GenericArg::Lifetime(lt) if lt.is_elided() => true,
                    GenericArg::Lifetime(_) => {
                        nonelided_generic_args = true;
                        false
                    }
                    _ => {
                        nonelided_generic_args = true;
                        true
                    }
                });

                if nonelided_generic_args {
                    start_or_comma(self);
                    self.commasep(Inconsistent, generic_args.args, |s, generic_arg| {
                        s.print_generic_arg(generic_arg, elide_lifetimes)
                    });
                }

                for constraint in generic_args.constraints {
                    start_or_comma(self);
                    self.print_assoc_item_constraint(constraint);
                }

                if !empty.get() {
                    self.word(">")
                }
            }
            hir::GenericArgsParentheses::ParenSugar => {
                let (inputs, output) = generic_args.paren_sugar_inputs_output().unwrap();

                self.word("(");
                self.commasep(Inconsistent, inputs, |s, ty| s.print_type(ty));
                self.word(")");

                self.space_if_not_bol();
                self.word_space("->");
                self.print_type(output);
            }
            hir::GenericArgsParentheses::ReturnTypeNotation => {
                self.word("(..)");
            }
        }
    }

    fn print_assoc_item_constraint(&mut self, constraint: &hir::AssocItemConstraint<'_>) {
        self.print_ident(constraint.ident);
        self.print_generic_args(constraint.gen_args, false);
        self.space();
        match constraint.kind {
            hir::AssocItemConstraintKind::Equality { ref term } => {
                self.word_space("=");
                match term {
                    Term::Ty(ty) => self.print_type(ty),
                    Term::Const(c) => self.print_const_arg(c),
                }
            }
            hir::AssocItemConstraintKind::Bound { bounds } => {
                self.print_bounds(":", bounds);
            }
        }
    }

    fn print_pat_expr(&mut self, expr: &hir::PatExpr<'_>) {
        match &expr.kind {
            hir::PatExprKind::Lit { lit, negated } => {
                if *negated {
                    self.word("-");
                }
                self.print_literal(lit);
            }
            hir::PatExprKind::ConstBlock(c) => self.print_inline_const(c),
            hir::PatExprKind::Path(qpath) => self.print_qpath(qpath, true),
        }
    }

    fn print_ty_pat(&mut self, pat: &hir::TyPat<'_>) {
        self.maybe_print_comment(pat.span.lo());
        self.ann.pre(self, AnnNode::TyPat(pat));
        // Pat isn't normalized, but the beauty of it
        // is that it doesn't matter
        match pat.kind {
            TyPatKind::Range(begin, end) => {
                self.print_const_arg(begin);
                self.word("..=");
                self.print_const_arg(end);
            }
            TyPatKind::Or(patterns) => {
                self.popen();
                let mut first = true;
                for pat in patterns {
                    if first {
                        first = false;
                    } else {
                        self.word(" | ");
                    }
                    self.print_ty_pat(pat);
                }
                self.pclose();
            }
            TyPatKind::Err(_) => {
                self.popen();
                self.word("/*ERROR*/");
                self.pclose();
            }
        }
        self.ann.post(self, AnnNode::TyPat(pat))
    }

    fn print_pat(&mut self, pat: &hir::Pat<'_>) {
        self.maybe_print_comment(pat.span.lo());
        self.ann.pre(self, AnnNode::Pat(pat));
        // Pat isn't normalized, but the beauty of it is that it doesn't matter.
        match pat.kind {
            // Printing `_` isn't ideal for a missing pattern, but it's easy and good enough.
            // E.g. `fn(u32)` gets printed as `fn(_: u32)`.
            PatKind::Missing => self.word("_"),
            PatKind::Wild => self.word("_"),
            PatKind::Never => self.word("!"),
            PatKind::Binding(BindingMode(by_ref, mutbl), _, ident, sub) => {
                if mutbl.is_mut() {
                    self.word_nbsp("mut");
                }
                if let ByRef::Yes(rmutbl) = by_ref {
                    self.word_nbsp("ref");
                    if rmutbl.is_mut() {
                        self.word_nbsp("mut");
                    }
                }
                self.print_ident(ident);
                if let Some(p) = sub {
                    self.word("@");
                    self.print_pat(p);
                }
            }
            PatKind::TupleStruct(ref qpath, elts, ddpos) => {
                self.print_qpath(qpath, true);
                self.popen();
                if let Some(ddpos) = ddpos.as_opt_usize() {
                    self.commasep(Inconsistent, &elts[..ddpos], |s, p| s.print_pat(p));
                    if ddpos != 0 {
                        self.word_space(",");
                    }
                    self.word("..");
                    if ddpos != elts.len() {
                        self.word(",");
                        self.commasep(Inconsistent, &elts[ddpos..], |s, p| s.print_pat(p));
                    }
                } else {
                    self.commasep(Inconsistent, elts, |s, p| s.print_pat(p));
                }
                self.pclose();
            }
            PatKind::Struct(ref qpath, fields, etc) => {
                self.print_qpath(qpath, true);
                self.nbsp();
                self.word("{");
                let empty = fields.is_empty() && !etc;
                if !empty {
                    self.space();
                }
                self.commasep_cmnt(Consistent, fields, |s, f| s.print_patfield(f), |f| f.pat.span);
                if etc {
                    if !fields.is_empty() {
                        self.word_space(",");
                    }
                    self.word("..");
                }
                if !empty {
                    self.space();
                }
                self.word("}");
            }
            PatKind::Or(pats) => {
                self.strsep("|", true, Inconsistent, pats, |s, p| s.print_pat(p));
            }
            PatKind::Tuple(elts, ddpos) => {
                self.popen();
                if let Some(ddpos) = ddpos.as_opt_usize() {
                    self.commasep(Inconsistent, &elts[..ddpos], |s, p| s.print_pat(p));
                    if ddpos != 0 {
                        self.word_space(",");
                    }
                    self.word("..");
                    if ddpos != elts.len() {
                        self.word(",");
                        self.commasep(Inconsistent, &elts[ddpos..], |s, p| s.print_pat(p));
                    }
                } else {
                    self.commasep(Inconsistent, elts, |s, p| s.print_pat(p));
                    if elts.len() == 1 {
                        self.word(",");
                    }
                }
                self.pclose();
            }
            PatKind::Box(inner) => {
                let is_range_inner = matches!(inner.kind, PatKind::Range(..));
                self.word("box ");
                if is_range_inner {
                    self.popen();
                }
                self.print_pat(inner);
                if is_range_inner {
                    self.pclose();
                }
            }
            PatKind::Deref(inner) => {
                self.word("deref!");
                self.popen();
                self.print_pat(inner);
                self.pclose();
            }
            PatKind::Ref(inner, mutbl) => {
                let is_range_inner = matches!(inner.kind, PatKind::Range(..));
                self.word("&");
                self.word(mutbl.prefix_str());
                if is_range_inner {
                    self.popen();
                }
                self.print_pat(inner);
                if is_range_inner {
                    self.pclose();
                }
            }
            PatKind::Expr(e) => self.print_pat_expr(e),
            PatKind::Range(begin, end, end_kind) => {
                if let Some(expr) = begin {
                    self.print_pat_expr(expr);
                }
                match end_kind {
                    RangeEnd::Included => self.word("..."),
                    RangeEnd::Excluded => self.word(".."),
                }
                if let Some(expr) = end {
                    self.print_pat_expr(expr);
                }
            }
            PatKind::Slice(before, slice, after) => {
                self.word("[");
                self.commasep(Inconsistent, before, |s, p| s.print_pat(p));
                if let Some(p) = slice {
                    if !before.is_empty() {
                        self.word_space(",");
                    }
                    if let PatKind::Wild = p.kind {
                        // Print nothing.
                    } else {
                        self.print_pat(p);
                    }
                    self.word("..");
                    if !after.is_empty() {
                        self.word_space(",");
                    }
                }
                self.commasep(Inconsistent, after, |s, p| s.print_pat(p));
                self.word("]");
            }
            PatKind::Guard(inner, cond) => {
                self.print_pat(inner);
                self.space();
                self.word_space("if");
                self.print_expr(cond);
            }
            PatKind::Err(_) => {
                self.popen();
                self.word("/*ERROR*/");
                self.pclose();
            }
        }
        self.ann.post(self, AnnNode::Pat(pat))
    }

    fn print_patfield(&mut self, field: &hir::PatField<'_>) {
        if self.attrs(field.hir_id).is_empty() {
            self.space();
        }
        let cb = self.cbox(INDENT_UNIT);
        self.print_attrs_as_outer(self.attrs(field.hir_id));
        if !field.is_shorthand {
            self.print_ident(field.ident);
            self.word_nbsp(":");
        }
        self.print_pat(field.pat);
        self.end(cb);
    }

    fn print_param(&mut self, arg: &hir::Param<'_>) {
        self.print_attrs_as_outer(self.attrs(arg.hir_id));
        self.print_pat(arg.pat);
    }

    fn print_implicit_self(&mut self, implicit_self_kind: &hir::ImplicitSelfKind) {
        match implicit_self_kind {
            ImplicitSelfKind::Imm => {
                self.word("self");
            }
            ImplicitSelfKind::Mut => {
                self.print_mutability(hir::Mutability::Mut, false);
                self.word("self");
            }
            ImplicitSelfKind::RefImm => {
                self.word("&");
                self.word("self");
            }
            ImplicitSelfKind::RefMut => {
                self.word("&");
                self.print_mutability(hir::Mutability::Mut, false);
                self.word("self");
            }
            ImplicitSelfKind::None => unreachable!(),
        }
    }

    fn print_arm(&mut self, arm: &hir::Arm<'_>) {
        // I have no idea why this check is necessary, but here it
        // is :(
        if self.attrs(arm.hir_id).is_empty() {
            self.space();
        }
        let cb = self.cbox(INDENT_UNIT);
        self.ann.pre(self, AnnNode::Arm(arm));
        let ib = self.ibox(0);
        self.print_attrs_as_outer(self.attrs(arm.hir_id));
        self.print_pat(arm.pat);
        self.space();
        if let Some(ref g) = arm.guard {
            self.word_space("if");
            self.print_expr(g);
            self.space();
        }
        self.word_space("=>");

        match arm.body.kind {
            hir::ExprKind::Block(blk, opt_label) => {
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }
                self.print_block_unclosed(blk, ib);

                // If it is a user-provided unsafe block, print a comma after it
                if let hir::BlockCheckMode::UnsafeBlock(hir::UnsafeSource::UserProvided) = blk.rules
                {
                    self.word(",");
                }
            }
            _ => {
                self.end(ib);
                self.print_expr(arm.body);
                self.word(",");
            }
        }
        self.ann.post(self, AnnNode::Arm(arm));
        self.end(cb)
    }

    fn print_fn(
        &mut self,
        header: hir::FnHeader,
        name: Option<Symbol>,
        generics: &hir::Generics<'_>,
        decl: &hir::FnDecl<'_>,
        arg_idents: &[Option<Ident>],
        body_id: Option<hir::BodyId>,
    ) {
        self.print_fn_header_info(header);

        if let Some(name) = name {
            self.nbsp();
            self.print_name(name);
        }
        self.print_generic_params(generics.params);

        self.popen();
        // Make sure we aren't supplied *both* `arg_idents` and `body_id`.
        assert!(arg_idents.is_empty() || body_id.is_none());
        let mut i = 0;
        let mut print_arg = |s: &mut Self, ty: Option<&hir::Ty<'_>>| {
            if i == 0 && decl.implicit_self.has_implicit_self() {
                s.print_implicit_self(&decl.implicit_self);
            } else {
                if let Some(arg_ident) = arg_idents.get(i) {
                    if let Some(arg_ident) = arg_ident {
                        s.word(arg_ident.to_string());
                        s.word(":");
                        s.space();
                    }
                } else if let Some(body_id) = body_id {
                    s.ann.nested(s, Nested::BodyParamPat(body_id, i));
                    s.word(":");
                    s.space();
                }
                if let Some(ty) = ty {
                    s.print_type(ty);
                }
            }
            i += 1;
        };
        self.commasep(Inconsistent, decl.inputs, |s, ty| {
            let ib = s.ibox(INDENT_UNIT);
            print_arg(s, Some(ty));
            s.end(ib);
        });
        if decl.c_variadic {
            if !decl.inputs.is_empty() {
                self.word(", ");
            }
            print_arg(self, None);
            self.word("...");
        }
        self.pclose();

        self.print_fn_output(decl);
        self.print_where_clause(generics)
    }

    fn print_closure_params(&mut self, decl: &hir::FnDecl<'_>, body_id: hir::BodyId) {
        self.word("|");
        let mut i = 0;
        self.commasep(Inconsistent, decl.inputs, |s, ty| {
            let ib = s.ibox(INDENT_UNIT);

            s.ann.nested(s, Nested::BodyParamPat(body_id, i));
            i += 1;

            if let hir::TyKind::Infer(()) = ty.kind {
                // Print nothing.
            } else {
                s.word(":");
                s.space();
                s.print_type(ty);
            }
            s.end(ib);
        });
        self.word("|");

        match decl.output {
            hir::FnRetTy::Return(ty) => {
                self.space_if_not_bol();
                self.word_space("->");
                self.print_type(ty);
                self.maybe_print_comment(ty.span.lo());
            }
            hir::FnRetTy::DefaultReturn(..) => {}
        }
    }

    fn print_capture_clause(&mut self, capture_clause: hir::CaptureBy) {
        match capture_clause {
            hir::CaptureBy::Value { .. } => self.word_space("move"),
            hir::CaptureBy::Use { .. } => self.word_space("use"),
            hir::CaptureBy::Ref => {}
        }
    }

    fn print_closure_binder(
        &mut self,
        binder: hir::ClosureBinder,
        generic_params: &[GenericParam<'_>],
    ) {
        let generic_params = generic_params
            .iter()
            .filter(|p| {
                matches!(
                    p,
                    GenericParam {
                        kind: GenericParamKind::Lifetime { kind: LifetimeParamKind::Explicit },
                        ..
                    }
                )
            })
            .collect::<Vec<_>>();

        match binder {
            hir::ClosureBinder::Default => {}
            // We need to distinguish `|...| {}` from `for<> |...| {}` as `for<>` adds additional
            // restrictions.
            hir::ClosureBinder::For { .. } if generic_params.is_empty() => self.word("for<>"),
            hir::ClosureBinder::For { .. } => {
                self.word("for");
                self.word("<");

                self.commasep(Inconsistent, &generic_params, |s, param| {
                    s.print_generic_param(param)
                });

                self.word(">");
                self.nbsp();
            }
        }
    }

    fn print_bounds<'b>(
        &mut self,
        prefix: &'static str,
        bounds: impl IntoIterator<Item = &'b hir::GenericBound<'b>>,
    ) {
        let mut first = true;
        for bound in bounds {
            if first {
                self.word(prefix);
            }
            if !(first && prefix.is_empty()) {
                self.nbsp();
            }
            if first {
                first = false;
            } else {
                self.word_space("+");
            }

            match bound {
                GenericBound::Trait(tref) => {
                    self.print_poly_trait_ref(tref);
                }
                GenericBound::Outlives(lt) => {
                    self.print_lifetime(lt);
                }
                GenericBound::Use(args, _) => {
                    self.word("use <");

                    self.commasep(Inconsistent, *args, |s, arg| {
                        s.print_precise_capturing_arg(*arg)
                    });

                    self.word(">");
                }
            }
        }
    }

    fn print_precise_capturing_arg(&mut self, arg: PreciseCapturingArg<'_>) {
        match arg {
            PreciseCapturingArg::Lifetime(lt) => self.print_lifetime(lt),
            PreciseCapturingArg::Param(arg) => self.print_ident(arg.ident),
        }
    }

    fn print_generic_params(&mut self, generic_params: &[GenericParam<'_>]) {
        let is_lifetime_elided = |generic_param: &GenericParam<'_>| {
            matches!(
                generic_param.kind,
                GenericParamKind::Lifetime { kind: LifetimeParamKind::Elided(_) }
            )
        };

        // We don't want to show elided lifetimes as they are compiler-inserted and not
        // expressible in surface level Rust.
        if !generic_params.is_empty() && !generic_params.iter().all(is_lifetime_elided) {
            self.word("<");

            self.commasep(
                Inconsistent,
                generic_params.iter().filter(|gp| !is_lifetime_elided(gp)),
                |s, param| s.print_generic_param(param),
            );

            self.word(">");
        }
    }

    fn print_generic_param(&mut self, param: &GenericParam<'_>) {
        if let GenericParamKind::Const { .. } = param.kind {
            self.word_space("const");
        }

        self.print_ident(param.name.ident());

        match param.kind {
            GenericParamKind::Lifetime { .. } => {}
            GenericParamKind::Type { default, .. } => {
                if let Some(default) = default {
                    self.space();
                    self.word_space("=");
                    self.print_type(default);
                }
            }
            GenericParamKind::Const { ty, ref default, synthetic: _ } => {
                self.word_space(":");
                self.print_type(ty);
                if let Some(default) = default {
                    self.space();
                    self.word_space("=");
                    self.print_const_arg(default);
                }
            }
        }
    }

    fn print_lifetime(&mut self, lifetime: &hir::Lifetime) {
        self.print_ident(lifetime.ident)
    }

    fn print_where_clause(&mut self, generics: &hir::Generics<'_>) {
        if generics.predicates.is_empty() {
            return;
        }

        self.space();
        self.word_space("where");

        for (i, predicate) in generics.predicates.iter().enumerate() {
            if i != 0 {
                self.word_space(",");
            }
            self.print_where_predicate(predicate);
        }
    }

    fn print_where_predicate(&mut self, predicate: &hir::WherePredicate<'_>) {
        self.print_attrs_as_outer(self.attrs(predicate.hir_id));
        match *predicate.kind {
            hir::WherePredicateKind::BoundPredicate(hir::WhereBoundPredicate {
                bound_generic_params,
                bounded_ty,
                bounds,
                ..
            }) => {
                self.print_formal_generic_params(bound_generic_params);
                self.print_type(bounded_ty);
                self.print_bounds(":", bounds);
            }
            hir::WherePredicateKind::RegionPredicate(hir::WhereRegionPredicate {
                lifetime,
                bounds,
                ..
            }) => {
                self.print_lifetime(lifetime);
                self.word(":");

                for (i, bound) in bounds.iter().enumerate() {
                    match bound {
                        GenericBound::Outlives(lt) => {
                            self.print_lifetime(lt);
                        }
                        _ => panic!("unexpected bound on lifetime param: {bound:?}"),
                    }

                    if i != 0 {
                        self.word(":");
                    }
                }
            }
            hir::WherePredicateKind::EqPredicate(hir::WhereEqPredicate {
                lhs_ty, rhs_ty, ..
            }) => {
                self.print_type(lhs_ty);
                self.space();
                self.word_space("=");
                self.print_type(rhs_ty);
            }
        }
    }

    fn print_mutability(&mut self, mutbl: hir::Mutability, print_const: bool) {
        match mutbl {
            hir::Mutability::Mut => self.word_nbsp("mut"),
            hir::Mutability::Not => {
                if print_const {
                    self.word_nbsp("const")
                }
            }
        }
    }

    fn print_mt(&mut self, mt: &hir::MutTy<'_>, print_const: bool) {
        self.print_mutability(mt.mutbl, print_const);
        self.print_type(mt.ty);
    }

    fn print_fn_output(&mut self, decl: &hir::FnDecl<'_>) {
        match decl.output {
            hir::FnRetTy::Return(ty) => {
                self.space_if_not_bol();
                let ib = self.ibox(INDENT_UNIT);
                self.word_space("->");
                self.print_type(ty);
                self.end(ib);

                if let hir::FnRetTy::Return(output) = decl.output {
                    self.maybe_print_comment(output.span.lo());
                }
            }
            hir::FnRetTy::DefaultReturn(..) => {}
        }
    }

    fn print_ty_fn(
        &mut self,
        abi: ExternAbi,
        safety: hir::Safety,
        decl: &hir::FnDecl<'_>,
        name: Option<Symbol>,
        generic_params: &[hir::GenericParam<'_>],
        arg_idents: &[Option<Ident>],
    ) {
        let ib = self.ibox(INDENT_UNIT);
        self.print_formal_generic_params(generic_params);
        let generics = hir::Generics::empty();
        self.print_fn(
            hir::FnHeader {
                safety: safety.into(),
                abi,
                constness: hir::Constness::NotConst,
                asyncness: hir::IsAsync::NotAsync,
            },
            name,
            generics,
            decl,
            arg_idents,
            None,
        );
        self.end(ib);
    }

    fn print_fn_header_info(&mut self, header: hir::FnHeader) {
        self.print_constness(header.constness);

        let safety = match header.safety {
            hir::HeaderSafety::SafeTargetFeatures => {
                self.word_nbsp("#[target_feature]");
                hir::Safety::Safe
            }
            hir::HeaderSafety::Normal(safety) => safety,
        };

        match header.asyncness {
            hir::IsAsync::NotAsync => {}
            hir::IsAsync::Async(_) => self.word_nbsp("async"),
        }

        self.print_safety(safety);

        if header.abi != ExternAbi::Rust {
            self.word_nbsp("extern");
            self.word_nbsp(header.abi.to_string());
        }

        self.word("fn")
    }

    fn print_constness(&mut self, s: hir::Constness) {
        match s {
            hir::Constness::NotConst => {}
            hir::Constness::Const => self.word_nbsp("const"),
        }
    }

    fn print_safety(&mut self, s: hir::Safety) {
        match s {
            hir::Safety::Safe => {}
            hir::Safety::Unsafe => self.word_nbsp("unsafe"),
        }
    }

    fn print_is_auto(&mut self, s: hir::IsAuto) {
        match s {
            hir::IsAuto::Yes => self.word_nbsp("auto"),
            hir::IsAuto::No => {}
        }
    }
}

/// Does this expression require a semicolon to be treated
/// as a statement? The negation of this: 'can this expression
/// be used as a statement without a semicolon' -- is used
/// as an early-bail-out in the parser so that, for instance,
///     if true {...} else {...}
///      |x| 5
/// isn't parsed as (if true {...} else {...} | x) | 5
//
// Duplicated from `parse::classify`, but adapted for the HIR.
fn expr_requires_semi_to_be_stmt(e: &hir::Expr<'_>) -> bool {
    !matches!(
        e.kind,
        hir::ExprKind::If(..)
            | hir::ExprKind::Match(..)
            | hir::ExprKind::Block(..)
            | hir::ExprKind::Loop(..)
    )
}

/// This statement requires a semicolon after it.
/// note that in one case (stmt_semi), we've already
/// seen the semicolon, and thus don't need another.
fn stmt_ends_with_semi(stmt: &hir::StmtKind<'_>) -> bool {
    match *stmt {
        hir::StmtKind::Let(_) => true,
        hir::StmtKind::Item(_) => false,
        hir::StmtKind::Expr(e) => expr_requires_semi_to_be_stmt(e),
        hir::StmtKind::Semi(..) => false,
    }
}

/// Expressions that syntactically contain an "exterior" struct literal, i.e., not surrounded by any
/// parens or other delimiters, e.g., `X { y: 1 }`, `X { y: 1 }.method()`, `foo == X { y: 1 }` and
/// `X { y: 1 } == foo` all do, but `(X { y: 1 }) == foo` does not.
fn contains_exterior_struct_lit(value: &hir::Expr<'_>) -> bool {
    match value.kind {
        hir::ExprKind::Struct(..) => true,

        hir::ExprKind::Assign(lhs, rhs, _)
        | hir::ExprKind::AssignOp(_, lhs, rhs)
        | hir::ExprKind::Binary(_, lhs, rhs) => {
            // `X { y: 1 } + X { y: 2 }`
            contains_exterior_struct_lit(lhs) || contains_exterior_struct_lit(rhs)
        }
        hir::ExprKind::Unary(_, x)
        | hir::ExprKind::Cast(x, _)
        | hir::ExprKind::Type(x, _)
        | hir::ExprKind::Field(x, _)
        | hir::ExprKind::Index(x, _, _) => {
            // `&X { y: 1 }, X { y: 1 }.y`
            contains_exterior_struct_lit(x)
        }

        hir::ExprKind::MethodCall(_, receiver, ..) => {
            // `X { y: 1 }.bar(...)`
            contains_exterior_struct_lit(receiver)
        }

        _ => false,
    }
}
