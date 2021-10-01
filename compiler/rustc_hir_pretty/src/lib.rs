#![recursion_limit = "256"]

use rustc_ast as ast;
use rustc_ast::util::parser::{self, AssocOp, Fixity};
use rustc_ast_pretty::pp::Breaks::{Consistent, Inconsistent};
use rustc_ast_pretty::pp::{self, Breaks};
use rustc_ast_pretty::pprust::{Comments, PrintState};
use rustc_hir as hir;
use rustc_hir::{GenericArg, GenericParam, GenericParamKind, Node};
use rustc_hir::{GenericBound, PatKind, RangeEnd, TraitBoundModifier};
use rustc_span::source_map::{SourceMap, Spanned};
use rustc_span::symbol::{kw, Ident, IdentPrinter, Symbol};
use rustc_span::{self, BytePos, FileName};
use rustc_target::spec::abi::Abi;

use std::borrow::Cow;
use std::cell::Cell;
use std::collections::BTreeMap;
use std::vec;

pub fn id_to_string(map: &dyn rustc_hir::intravisit::Map<'_>, hir_id: hir::HirId) -> String {
    to_string(&map, |s| s.print_node(map.find(hir_id).unwrap()))
}

pub enum AnnNode<'a> {
    Name(&'a Symbol),
    Block(&'a hir::Block<'a>),
    Item(&'a hir::Item<'a>),
    SubItem(hir::HirId),
    Expr(&'a hir::Expr<'a>),
    Pat(&'a hir::Pat<'a>),
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

pub struct NoAnn;
impl PpAnn for NoAnn {}
pub const NO_ANN: &dyn PpAnn = &NoAnn;

/// Identical to the `PpAnn` implementation for `hir::Crate`,
/// except it avoids creating a dependency on the whole crate.
impl PpAnn for &dyn rustc_hir::intravisit::Map<'_> {
    fn nested(&self, state: &mut State<'_>, nested: Nested) {
        match nested {
            Nested::Item(id) => state.print_item(self.item(id)),
            Nested::TraitItem(id) => state.print_trait_item(self.trait_item(id)),
            Nested::ImplItem(id) => state.print_impl_item(self.impl_item(id)),
            Nested::ForeignItem(id) => state.print_foreign_item(self.foreign_item(id)),
            Nested::Body(id) => state.print_expr(&self.body(id).value),
            Nested::BodyParamPat(id, i) => state.print_pat(&self.body(id).params[i].pat),
        }
    }
}

pub struct State<'a> {
    pub s: pp::Printer,
    comments: Option<Comments<'a>>,
    attrs: &'a BTreeMap<hir::HirId, &'a [ast::Attribute]>,
    ann: &'a (dyn PpAnn + 'a),
}

impl<'a> State<'a> {
    pub fn print_node(&mut self, node: Node<'_>) {
        match node {
            Node::Param(a) => self.print_param(&a),
            Node::Item(a) => self.print_item(&a),
            Node::ForeignItem(a) => self.print_foreign_item(&a),
            Node::TraitItem(a) => self.print_trait_item(a),
            Node::ImplItem(a) => self.print_impl_item(a),
            Node::Variant(a) => self.print_variant(&a),
            Node::AnonConst(a) => self.print_anon_const(&a),
            Node::Expr(a) => self.print_expr(&a),
            Node::Stmt(a) => self.print_stmt(&a),
            Node::PathSegment(a) => self.print_path_segment(&a),
            Node::Ty(a) => self.print_type(&a),
            Node::TraitRef(a) => self.print_trait_ref(&a),
            Node::Binding(a) | Node::Pat(a) => self.print_pat(&a),
            Node::Arm(a) => self.print_arm(&a),
            Node::Infer(_) => self.s.word("_"),
            Node::Block(a) => {
                // Containing cbox, will be closed by print-block at `}`.
                self.cbox(INDENT_UNIT);
                // Head-ibox, will be closed by print-block after `{`.
                self.ibox(0);
                self.print_block(&a)
            }
            Node::Lifetime(a) => self.print_lifetime(&a),
            Node::Visibility(a) => self.print_visibility(&a),
            Node::GenericParam(_) => panic!("cannot print Node::GenericParam"),
            Node::Field(_) => panic!("cannot print Node::Field"),
            // These cases do not carry enough information in the
            // `hir_map` to reconstruct their full structure for pretty
            // printing.
            Node::Ctor(..) => panic!("cannot print isolated Ctor"),
            Node::Local(a) => self.print_local_decl(&a),
            Node::Crate(..) => panic!("cannot print Crate"),
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
    fn comments(&mut self) -> &mut Option<Comments<'a>> {
        &mut self.comments
    }

    fn print_ident(&mut self, ident: Ident) {
        self.s.word(IdentPrinter::for_ast_ident(ident, ident.is_raw_guess()).to_string());
        self.ann.post(self, AnnNode::Name(&ident.name))
    }

    fn print_generic_args(&mut self, _: &ast::GenericArgs, _colons_before_params: bool) {
        panic!("AST generic args printed by HIR pretty-printer");
    }
}

pub const INDENT_UNIT: usize = 4;

/// Requires you to pass an input filename and reader so that
/// it can scan the input text for comments to copy forward.
pub fn print_crate<'a>(
    sm: &'a SourceMap,
    krate: &hir::Crate<'_>,
    filename: FileName,
    input: String,
    ann: &'a dyn PpAnn,
) -> String {
    let mut s = State::new_from_input(sm, filename, input, &krate.attrs, ann);

    // When printing the AST, we sometimes need to inject `#[no_std]` here.
    // Since you can't compile the HIR, it's not necessary.

    s.print_mod(&krate.module(), s.attrs(hir::CRATE_HIR_ID));
    s.print_remaining_comments();
    s.s.eof()
}

impl<'a> State<'a> {
    pub fn new_from_input(
        sm: &'a SourceMap,
        filename: FileName,
        input: String,
        attrs: &'a BTreeMap<hir::HirId, &[ast::Attribute]>,
        ann: &'a dyn PpAnn,
    ) -> State<'a> {
        State {
            s: pp::mk_printer(),
            comments: Some(Comments::new(sm, filename, input)),
            attrs,
            ann,
        }
    }

    fn attrs(&self, id: hir::HirId) -> &'a [ast::Attribute] {
        self.attrs.get(&id).map_or(&[], |la| *la)
    }
}

pub fn to_string<F>(ann: &dyn PpAnn, f: F) -> String
where
    F: FnOnce(&mut State<'_>),
{
    let mut printer =
        State { s: pp::mk_printer(), comments: None, attrs: &BTreeMap::default(), ann };
    f(&mut printer);
    printer.s.eof()
}

pub fn visibility_qualified<S: Into<Cow<'static, str>>>(vis: &hir::Visibility<'_>, w: S) -> String {
    to_string(NO_ANN, |s| {
        s.print_visibility(vis);
        s.s.word(w)
    })
}

pub fn generic_params_to_string(generic_params: &[GenericParam<'_>]) -> String {
    to_string(NO_ANN, |s| s.print_generic_params(generic_params))
}

pub fn bounds_to_string<'b>(bounds: impl IntoIterator<Item = &'b hir::GenericBound<'b>>) -> String {
    to_string(NO_ANN, |s| s.print_bounds("", bounds))
}

pub fn ty_to_string(ty: &hir::Ty<'_>) -> String {
    to_string(NO_ANN, |s| s.print_type(ty))
}

pub fn path_segment_to_string(segment: &hir::PathSegment<'_>) -> String {
    to_string(NO_ANN, |s| s.print_path_segment(segment))
}

pub fn path_to_string(segment: &hir::Path<'_>) -> String {
    to_string(NO_ANN, |s| s.print_path(segment, false))
}

pub fn fn_to_string(
    decl: &hir::FnDecl<'_>,
    header: hir::FnHeader,
    name: Option<Symbol>,
    generics: &hir::Generics<'_>,
    vis: &hir::Visibility<'_>,
    arg_names: &[Ident],
    body_id: Option<hir::BodyId>,
) -> String {
    to_string(NO_ANN, |s| s.print_fn(decl, header, name, generics, vis, arg_names, body_id))
}

pub fn enum_def_to_string(
    enum_definition: &hir::EnumDef<'_>,
    generics: &hir::Generics<'_>,
    name: Symbol,
    span: rustc_span::Span,
    visibility: &hir::Visibility<'_>,
) -> String {
    to_string(NO_ANN, |s| s.print_enum_def(enum_definition, generics, name, span, visibility))
}

impl<'a> State<'a> {
    pub fn cbox(&mut self, u: usize) {
        self.s.cbox(u);
    }

    pub fn nbsp(&mut self) {
        self.s.word(" ")
    }

    pub fn word_nbsp<S: Into<Cow<'static, str>>>(&mut self, w: S) {
        self.s.word(w);
        self.nbsp()
    }

    pub fn head<S: Into<Cow<'static, str>>>(&mut self, w: S) {
        let w = w.into();
        // outer-box is consistent
        self.cbox(INDENT_UNIT);
        // head-box is inconsistent
        self.ibox(w.len() + 1);
        // keyword that starts the head
        if !w.is_empty() {
            self.word_nbsp(w);
        }
    }

    pub fn bopen(&mut self) {
        self.s.word("{");
        self.end(); // close the head-box
    }

    pub fn bclose_maybe_open(&mut self, span: rustc_span::Span, close_box: bool) {
        self.maybe_print_comment(span.hi());
        self.break_offset_if_not_bol(1, -(INDENT_UNIT as isize));
        self.s.word("}");
        if close_box {
            self.end(); // close the outer-box
        }
    }

    pub fn bclose(&mut self, span: rustc_span::Span) {
        self.bclose_maybe_open(span, true)
    }

    pub fn space_if_not_bol(&mut self) {
        if !self.s.is_beginning_of_line() {
            self.s.space();
        }
    }

    pub fn break_offset_if_not_bol(&mut self, n: usize, off: isize) {
        if !self.s.is_beginning_of_line() {
            self.s.break_offset(n, off)
        } else if off != 0 && self.s.last_token().is_hardbreak_tok() {
            // We do something pretty sketchy here: tuck the nonzero
            // offset-adjustment we were going to deposit along with the
            // break into the previous hardbreak.
            self.s.replace_last_token(pp::Printer::hardbreak_tok_offset(off));
        }
    }

    // Synthesizes a comment that was not textually present in the original source
    // file.
    pub fn synth_comment(&mut self, text: String) {
        self.s.word("/*");
        self.s.space();
        self.s.word(text);
        self.s.space();
        self.s.word("*/")
    }

    pub fn commasep_cmnt<T, F, G>(&mut self, b: Breaks, elts: &[T], mut op: F, mut get_span: G)
    where
        F: FnMut(&mut State<'_>, &T),
        G: FnMut(&T) -> rustc_span::Span,
    {
        self.rbox(0, b);
        let len = elts.len();
        let mut i = 0;
        for elt in elts {
            self.maybe_print_comment(get_span(elt).hi());
            op(self, elt);
            i += 1;
            if i < len {
                self.s.word(",");
                self.maybe_print_trailing_comment(get_span(elt), Some(get_span(&elts[i]).hi()));
                self.space_if_not_bol();
            }
        }
        self.end();
    }

    pub fn commasep_exprs(&mut self, b: Breaks, exprs: &[hir::Expr<'_>]) {
        self.commasep_cmnt(b, exprs, |s, e| s.print_expr(&e), |e| e.span)
    }

    pub fn print_mod(&mut self, _mod: &hir::Mod<'_>, attrs: &[ast::Attribute]) {
        self.print_inner_attributes(attrs);
        for &item_id in _mod.item_ids {
            self.ann.nested(self, Nested::Item(item_id));
        }
    }

    pub fn print_opt_lifetime(&mut self, lifetime: &hir::Lifetime) {
        if !lifetime.is_elided() {
            self.print_lifetime(lifetime);
            self.nbsp();
        }
    }

    pub fn print_type(&mut self, ty: &hir::Ty<'_>) {
        self.maybe_print_comment(ty.span.lo());
        self.ibox(0);
        match ty.kind {
            hir::TyKind::Slice(ref ty) => {
                self.s.word("[");
                self.print_type(&ty);
                self.s.word("]");
            }
            hir::TyKind::Ptr(ref mt) => {
                self.s.word("*");
                self.print_mt(mt, true);
            }
            hir::TyKind::Rptr(ref lifetime, ref mt) => {
                self.s.word("&");
                self.print_opt_lifetime(lifetime);
                self.print_mt(mt, false);
            }
            hir::TyKind::Never => {
                self.s.word("!");
            }
            hir::TyKind::Tup(ref elts) => {
                self.popen();
                self.commasep(Inconsistent, &elts[..], |s, ty| s.print_type(&ty));
                if elts.len() == 1 {
                    self.s.word(",");
                }
                self.pclose();
            }
            hir::TyKind::BareFn(ref f) => {
                self.print_ty_fn(
                    f.abi,
                    f.unsafety,
                    &f.decl,
                    None,
                    &f.generic_params,
                    f.param_names,
                );
            }
            hir::TyKind::OpaqueDef(..) => self.s.word("/*impl Trait*/"),
            hir::TyKind::Path(ref qpath) => self.print_qpath(qpath, false),
            hir::TyKind::TraitObject(bounds, ref lifetime, syntax) => {
                if syntax == ast::TraitObjectSyntax::Dyn {
                    self.word_space("dyn");
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
                    self.print_lifetime(lifetime);
                }
            }
            hir::TyKind::Array(ref ty, ref length) => {
                self.s.word("[");
                self.print_type(&ty);
                self.s.word("; ");
                self.print_anon_const(length);
                self.s.word("]");
            }
            hir::TyKind::Typeof(ref e) => {
                self.s.word("typeof(");
                self.print_anon_const(e);
                self.s.word(")");
            }
            hir::TyKind::Err => {
                self.popen();
                self.s.word("/*ERROR*/");
                self.pclose();
            }
            hir::TyKind::Infer => {
                self.s.word("_");
            }
        }
        self.end()
    }

    pub fn print_foreign_item(&mut self, item: &hir::ForeignItem<'_>) {
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(item.span.lo());
        self.print_outer_attributes(self.attrs(item.hir_id()));
        match item.kind {
            hir::ForeignItemKind::Fn(ref decl, ref arg_names, ref generics) => {
                self.head("");
                self.print_fn(
                    decl,
                    hir::FnHeader {
                        unsafety: hir::Unsafety::Normal,
                        constness: hir::Constness::NotConst,
                        abi: Abi::Rust,
                        asyncness: hir::IsAsync::NotAsync,
                    },
                    Some(item.ident.name),
                    generics,
                    &item.vis,
                    arg_names,
                    None,
                );
                self.end(); // end head-ibox
                self.s.word(";");
                self.end() // end the outer fn box
            }
            hir::ForeignItemKind::Static(ref t, m) => {
                self.head(visibility_qualified(&item.vis, "static"));
                if m == hir::Mutability::Mut {
                    self.word_space("mut");
                }
                self.print_ident(item.ident);
                self.word_space(":");
                self.print_type(&t);
                self.s.word(";");
                self.end(); // end the head-ibox
                self.end() // end the outer cbox
            }
            hir::ForeignItemKind::Type => {
                self.head(visibility_qualified(&item.vis, "type"));
                self.print_ident(item.ident);
                self.s.word(";");
                self.end(); // end the head-ibox
                self.end() // end the outer cbox
            }
        }
    }

    fn print_associated_const(
        &mut self,
        ident: Ident,
        ty: &hir::Ty<'_>,
        default: Option<hir::BodyId>,
        vis: &hir::Visibility<'_>,
    ) {
        self.s.word(visibility_qualified(vis, ""));
        self.word_space("const");
        self.print_ident(ident);
        self.word_space(":");
        self.print_type(ty);
        if let Some(expr) = default {
            self.s.space();
            self.word_space("=");
            self.ann.nested(self, Nested::Body(expr));
        }
        self.s.word(";")
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
        self.print_generic_params(&generics.params);
        if let Some(bounds) = bounds {
            self.print_bounds(":", bounds);
        }
        self.print_where_clause(&generics.where_clause);
        if let Some(ty) = ty {
            self.s.space();
            self.word_space("=");
            self.print_type(ty);
        }
        self.s.word(";")
    }

    fn print_item_type(
        &mut self,
        item: &hir::Item<'_>,
        generics: &hir::Generics<'_>,
        inner: impl Fn(&mut Self),
    ) {
        self.head(visibility_qualified(&item.vis, "type"));
        self.print_ident(item.ident);
        self.print_generic_params(&generics.params);
        self.end(); // end the inner ibox

        self.print_where_clause(&generics.where_clause);
        self.s.space();
        inner(self);
        self.s.word(";");
        self.end(); // end the outer ibox
    }

    /// Pretty-print an item
    pub fn print_item(&mut self, item: &hir::Item<'_>) {
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(item.span.lo());
        let attrs = self.attrs(item.hir_id());
        self.print_outer_attributes(attrs);
        self.ann.pre(self, AnnNode::Item(item));
        match item.kind {
            hir::ItemKind::ExternCrate(orig_name) => {
                self.head(visibility_qualified(&item.vis, "extern crate"));
                if let Some(orig_name) = orig_name {
                    self.print_name(orig_name);
                    self.s.space();
                    self.s.word("as");
                    self.s.space();
                }
                self.print_ident(item.ident);
                self.s.word(";");
                self.end(); // end inner head-block
                self.end(); // end outer head-block
            }
            hir::ItemKind::Use(ref path, kind) => {
                self.head(visibility_qualified(&item.vis, "use"));
                self.print_path(path, false);

                match kind {
                    hir::UseKind::Single => {
                        if path.segments.last().unwrap().ident != item.ident {
                            self.s.space();
                            self.word_space("as");
                            self.print_ident(item.ident);
                        }
                        self.s.word(";");
                    }
                    hir::UseKind::Glob => self.s.word("::*;"),
                    hir::UseKind::ListStem => self.s.word("::{};"),
                }
                self.end(); // end inner head-block
                self.end(); // end outer head-block
            }
            hir::ItemKind::Static(ref ty, m, expr) => {
                self.head(visibility_qualified(&item.vis, "static"));
                if m == hir::Mutability::Mut {
                    self.word_space("mut");
                }
                self.print_ident(item.ident);
                self.word_space(":");
                self.print_type(&ty);
                self.s.space();
                self.end(); // end the head-ibox

                self.word_space("=");
                self.ann.nested(self, Nested::Body(expr));
                self.s.word(";");
                self.end(); // end the outer cbox
            }
            hir::ItemKind::Const(ref ty, expr) => {
                self.head(visibility_qualified(&item.vis, "const"));
                self.print_ident(item.ident);
                self.word_space(":");
                self.print_type(&ty);
                self.s.space();
                self.end(); // end the head-ibox

                self.word_space("=");
                self.ann.nested(self, Nested::Body(expr));
                self.s.word(";");
                self.end(); // end the outer cbox
            }
            hir::ItemKind::Fn(ref sig, ref param_names, body) => {
                self.head("");
                self.print_fn(
                    &sig.decl,
                    sig.header,
                    Some(item.ident.name),
                    param_names,
                    &item.vis,
                    &[],
                    Some(body),
                );
                self.s.word(" ");
                self.end(); // need to close a box
                self.end(); // need to close a box
                self.ann.nested(self, Nested::Body(body));
            }
            hir::ItemKind::Macro(ref macro_def) => {
                self.print_mac_def(macro_def, &item.ident, &item.span, |state| {
                    state.print_visibility(&item.vis)
                });
            }
            hir::ItemKind::Mod(ref _mod) => {
                self.head(visibility_qualified(&item.vis, "mod"));
                self.print_ident(item.ident);
                self.nbsp();
                self.bopen();
                self.print_mod(_mod, attrs);
                self.bclose(item.span);
            }
            hir::ItemKind::ForeignMod { abi, items } => {
                self.head("extern");
                self.word_nbsp(abi.to_string());
                self.bopen();
                self.print_inner_attributes(self.attrs(item.hir_id()));
                for item in items {
                    self.ann.nested(self, Nested::ForeignItem(item.id));
                }
                self.bclose(item.span);
            }
            hir::ItemKind::GlobalAsm(ref asm) => {
                self.head(visibility_qualified(&item.vis, "global_asm!"));
                self.print_inline_asm(asm);
                self.end()
            }
            hir::ItemKind::TyAlias(ref ty, ref generics) => {
                self.print_item_type(item, &generics, |state| {
                    state.word_space("=");
                    state.print_type(&ty);
                });
            }
            hir::ItemKind::OpaqueTy(ref opaque_ty) => {
                self.print_item_type(item, &opaque_ty.generics, |state| {
                    let mut real_bounds = Vec::with_capacity(opaque_ty.bounds.len());
                    for b in opaque_ty.bounds.iter() {
                        if let GenericBound::Trait(ref ptr, hir::TraitBoundModifier::Maybe) = *b {
                            state.s.space();
                            state.word_space("for ?");
                            state.print_trait_ref(&ptr.trait_ref);
                        } else {
                            real_bounds.push(b);
                        }
                    }
                    state.print_bounds("= impl", real_bounds);
                });
            }
            hir::ItemKind::Enum(ref enum_definition, ref params) => {
                self.print_enum_def(enum_definition, params, item.ident.name, item.span, &item.vis);
            }
            hir::ItemKind::Struct(ref struct_def, ref generics) => {
                self.head(visibility_qualified(&item.vis, "struct"));
                self.print_struct(struct_def, generics, item.ident.name, item.span, true);
            }
            hir::ItemKind::Union(ref struct_def, ref generics) => {
                self.head(visibility_qualified(&item.vis, "union"));
                self.print_struct(struct_def, generics, item.ident.name, item.span, true);
            }
            hir::ItemKind::Impl(hir::Impl {
                unsafety,
                polarity,
                defaultness,
                constness,
                defaultness_span: _,
                ref generics,
                ref of_trait,
                ref self_ty,
                items,
            }) => {
                self.head("");
                self.print_visibility(&item.vis);
                self.print_defaultness(defaultness);
                self.print_unsafety(unsafety);
                self.word_nbsp("impl");

                if !generics.params.is_empty() {
                    self.print_generic_params(&generics.params);
                    self.s.space();
                }

                if constness == hir::Constness::Const {
                    self.word_nbsp("const");
                }

                if let hir::ImplPolarity::Negative(_) = polarity {
                    self.s.word("!");
                }

                if let Some(ref t) = of_trait {
                    self.print_trait_ref(t);
                    self.s.space();
                    self.word_space("for");
                }

                self.print_type(&self_ty);
                self.print_where_clause(&generics.where_clause);

                self.s.space();
                self.bopen();
                self.print_inner_attributes(attrs);
                for impl_item in items {
                    self.ann.nested(self, Nested::ImplItem(impl_item.id));
                }
                self.bclose(item.span);
            }
            hir::ItemKind::Trait(is_auto, unsafety, ref generics, ref bounds, trait_items) => {
                self.head("");
                self.print_visibility(&item.vis);
                self.print_is_auto(is_auto);
                self.print_unsafety(unsafety);
                self.word_nbsp("trait");
                self.print_ident(item.ident);
                self.print_generic_params(&generics.params);
                let mut real_bounds = Vec::with_capacity(bounds.len());
                for b in bounds.iter() {
                    if let GenericBound::Trait(ref ptr, hir::TraitBoundModifier::Maybe) = *b {
                        self.s.space();
                        self.word_space("for ?");
                        self.print_trait_ref(&ptr.trait_ref);
                    } else {
                        real_bounds.push(b);
                    }
                }
                self.print_bounds(":", real_bounds);
                self.print_where_clause(&generics.where_clause);
                self.s.word(" ");
                self.bopen();
                for trait_item in trait_items {
                    self.ann.nested(self, Nested::TraitItem(trait_item.id));
                }
                self.bclose(item.span);
            }
            hir::ItemKind::TraitAlias(ref generics, ref bounds) => {
                self.head("");
                self.print_visibility(&item.vis);
                self.word_nbsp("trait");
                self.print_ident(item.ident);
                self.print_generic_params(&generics.params);
                let mut real_bounds = Vec::with_capacity(bounds.len());
                // FIXME(durka) this seems to be some quite outdated syntax
                for b in bounds.iter() {
                    if let GenericBound::Trait(ref ptr, hir::TraitBoundModifier::Maybe) = *b {
                        self.s.space();
                        self.word_space("for ?");
                        self.print_trait_ref(&ptr.trait_ref);
                    } else {
                        real_bounds.push(b);
                    }
                }
                self.nbsp();
                self.print_bounds("=", real_bounds);
                self.print_where_clause(&generics.where_clause);
                self.s.word(";");
            }
        }
        self.ann.post(self, AnnNode::Item(item))
    }

    pub fn print_trait_ref(&mut self, t: &hir::TraitRef<'_>) {
        self.print_path(&t.path, false)
    }

    fn print_formal_generic_params(&mut self, generic_params: &[hir::GenericParam<'_>]) {
        if !generic_params.is_empty() {
            self.s.word("for");
            self.print_generic_params(generic_params);
            self.nbsp();
        }
    }

    fn print_poly_trait_ref(&mut self, t: &hir::PolyTraitRef<'_>) {
        self.print_formal_generic_params(&t.bound_generic_params);
        self.print_trait_ref(&t.trait_ref)
    }

    pub fn print_enum_def(
        &mut self,
        enum_definition: &hir::EnumDef<'_>,
        generics: &hir::Generics<'_>,
        name: Symbol,
        span: rustc_span::Span,
        visibility: &hir::Visibility<'_>,
    ) {
        self.head(visibility_qualified(visibility, "enum"));
        self.print_name(name);
        self.print_generic_params(&generics.params);
        self.print_where_clause(&generics.where_clause);
        self.s.space();
        self.print_variants(&enum_definition.variants, span)
    }

    pub fn print_variants(&mut self, variants: &[hir::Variant<'_>], span: rustc_span::Span) {
        self.bopen();
        for v in variants {
            self.space_if_not_bol();
            self.maybe_print_comment(v.span.lo());
            self.print_outer_attributes(self.attrs(v.id));
            self.ibox(INDENT_UNIT);
            self.print_variant(v);
            self.s.word(",");
            self.end();
            self.maybe_print_trailing_comment(v.span, None);
        }
        self.bclose(span)
    }

    pub fn print_visibility(&mut self, vis: &hir::Visibility<'_>) {
        match vis.node {
            hir::VisibilityKind::Public => self.word_nbsp("pub"),
            hir::VisibilityKind::Crate(ast::CrateSugar::JustCrate) => self.word_nbsp("crate"),
            hir::VisibilityKind::Crate(ast::CrateSugar::PubCrate) => self.word_nbsp("pub(crate)"),
            hir::VisibilityKind::Restricted { ref path, .. } => {
                self.s.word("pub(");
                if path.segments.len() == 1 && path.segments[0].ident.name == kw::Super {
                    // Special case: `super` can print like `pub(super)`.
                    self.s.word("super");
                } else {
                    // Everything else requires `in` at present.
                    self.word_nbsp("in");
                    self.print_path(path, false);
                }
                self.word_nbsp(")");
            }
            hir::VisibilityKind::Inherited => (),
        }
    }

    pub fn print_defaultness(&mut self, defaultness: hir::Defaultness) {
        match defaultness {
            hir::Defaultness::Default { .. } => self.word_nbsp("default"),
            hir::Defaultness::Final => (),
        }
    }

    pub fn print_struct(
        &mut self,
        struct_def: &hir::VariantData<'_>,
        generics: &hir::Generics<'_>,
        name: Symbol,
        span: rustc_span::Span,
        print_finalizer: bool,
    ) {
        self.print_name(name);
        self.print_generic_params(&generics.params);
        match struct_def {
            hir::VariantData::Tuple(..) | hir::VariantData::Unit(..) => {
                if let hir::VariantData::Tuple(..) = struct_def {
                    self.popen();
                    self.commasep(Inconsistent, struct_def.fields(), |s, field| {
                        s.maybe_print_comment(field.span.lo());
                        s.print_outer_attributes(s.attrs(field.hir_id));
                        s.print_visibility(&field.vis);
                        s.print_type(&field.ty)
                    });
                    self.pclose();
                }
                self.print_where_clause(&generics.where_clause);
                if print_finalizer {
                    self.s.word(";");
                }
                self.end();
                self.end() // close the outer-box
            }
            hir::VariantData::Struct(..) => {
                self.print_where_clause(&generics.where_clause);
                self.nbsp();
                self.bopen();
                self.hardbreak_if_not_bol();

                for field in struct_def.fields() {
                    self.hardbreak_if_not_bol();
                    self.maybe_print_comment(field.span.lo());
                    self.print_outer_attributes(self.attrs(field.hir_id));
                    self.print_visibility(&field.vis);
                    self.print_ident(field.ident);
                    self.word_nbsp(":");
                    self.print_type(&field.ty);
                    self.s.word(",");
                }

                self.bclose(span)
            }
        }
    }

    pub fn print_variant(&mut self, v: &hir::Variant<'_>) {
        self.head("");
        let generics = hir::Generics::empty();
        self.print_struct(&v.data, &generics, v.ident.name, v.span, false);
        if let Some(ref d) = v.disr_expr {
            self.s.space();
            self.word_space("=");
            self.print_anon_const(d);
        }
    }
    pub fn print_method_sig(
        &mut self,
        ident: Ident,
        m: &hir::FnSig<'_>,
        generics: &hir::Generics<'_>,
        vis: &hir::Visibility<'_>,
        arg_names: &[Ident],
        body_id: Option<hir::BodyId>,
    ) {
        self.print_fn(&m.decl, m.header, Some(ident.name), generics, vis, arg_names, body_id)
    }

    pub fn print_trait_item(&mut self, ti: &hir::TraitItem<'_>) {
        self.ann.pre(self, AnnNode::SubItem(ti.hir_id()));
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(ti.span.lo());
        self.print_outer_attributes(self.attrs(ti.hir_id()));
        match ti.kind {
            hir::TraitItemKind::Const(ref ty, default) => {
                let vis =
                    Spanned { span: rustc_span::DUMMY_SP, node: hir::VisibilityKind::Inherited };
                self.print_associated_const(ti.ident, &ty, default, &vis);
            }
            hir::TraitItemKind::Fn(ref sig, hir::TraitFn::Required(ref arg_names)) => {
                let vis =
                    Spanned { span: rustc_span::DUMMY_SP, node: hir::VisibilityKind::Inherited };
                self.print_method_sig(ti.ident, sig, &ti.generics, &vis, arg_names, None);
                self.s.word(";");
            }
            hir::TraitItemKind::Fn(ref sig, hir::TraitFn::Provided(body)) => {
                let vis =
                    Spanned { span: rustc_span::DUMMY_SP, node: hir::VisibilityKind::Inherited };
                self.head("");
                self.print_method_sig(ti.ident, sig, &ti.generics, &vis, &[], Some(body));
                self.nbsp();
                self.end(); // need to close a box
                self.end(); // need to close a box
                self.ann.nested(self, Nested::Body(body));
            }
            hir::TraitItemKind::Type(ref bounds, ref default) => {
                self.print_associated_type(
                    ti.ident,
                    &ti.generics,
                    Some(bounds),
                    default.as_ref().map(|ty| &**ty),
                );
            }
        }
        self.ann.post(self, AnnNode::SubItem(ti.hir_id()))
    }

    pub fn print_impl_item(&mut self, ii: &hir::ImplItem<'_>) {
        self.ann.pre(self, AnnNode::SubItem(ii.hir_id()));
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(ii.span.lo());
        self.print_outer_attributes(self.attrs(ii.hir_id()));
        self.print_defaultness(ii.defaultness);

        match ii.kind {
            hir::ImplItemKind::Const(ref ty, expr) => {
                self.print_associated_const(ii.ident, &ty, Some(expr), &ii.vis);
            }
            hir::ImplItemKind::Fn(ref sig, body) => {
                self.head("");
                self.print_method_sig(ii.ident, sig, &ii.generics, &ii.vis, &[], Some(body));
                self.nbsp();
                self.end(); // need to close a box
                self.end(); // need to close a box
                self.ann.nested(self, Nested::Body(body));
            }
            hir::ImplItemKind::TyAlias(ref ty) => {
                self.print_associated_type(ii.ident, &ii.generics, None, Some(ty));
            }
        }
        self.ann.post(self, AnnNode::SubItem(ii.hir_id()))
    }

    pub fn print_local(&mut self, init: Option<&hir::Expr<'_>>, decl: impl Fn(&mut Self)) {
        self.space_if_not_bol();
        self.ibox(INDENT_UNIT);
        self.word_nbsp("let");

        self.ibox(INDENT_UNIT);
        decl(self);
        self.end();

        if let Some(ref init) = init {
            self.nbsp();
            self.word_space("=");
            self.print_expr(&init);
        }
        self.end()
    }

    pub fn print_stmt(&mut self, st: &hir::Stmt<'_>) {
        self.maybe_print_comment(st.span.lo());
        match st.kind {
            hir::StmtKind::Local(ref loc) => {
                self.print_local(loc.init, |this| this.print_local_decl(&loc));
            }
            hir::StmtKind::Item(item) => self.ann.nested(self, Nested::Item(item)),
            hir::StmtKind::Expr(ref expr) => {
                self.space_if_not_bol();
                self.print_expr(&expr);
            }
            hir::StmtKind::Semi(ref expr) => {
                self.space_if_not_bol();
                self.print_expr(&expr);
                self.s.word(";");
            }
        }
        if stmt_ends_with_semi(&st.kind) {
            self.s.word(";");
        }
        self.maybe_print_trailing_comment(st.span, None)
    }

    pub fn print_block(&mut self, blk: &hir::Block<'_>) {
        self.print_block_with_attrs(blk, &[])
    }

    pub fn print_block_unclosed(&mut self, blk: &hir::Block<'_>) {
        self.print_block_maybe_unclosed(blk, &[], false)
    }

    pub fn print_block_with_attrs(&mut self, blk: &hir::Block<'_>, attrs: &[ast::Attribute]) {
        self.print_block_maybe_unclosed(blk, attrs, true)
    }

    pub fn print_block_maybe_unclosed(
        &mut self,
        blk: &hir::Block<'_>,
        attrs: &[ast::Attribute],
        close_box: bool,
    ) {
        match blk.rules {
            hir::BlockCheckMode::UnsafeBlock(..) => self.word_space("unsafe"),
            hir::BlockCheckMode::DefaultBlock => (),
        }
        self.maybe_print_comment(blk.span.lo());
        self.ann.pre(self, AnnNode::Block(blk));
        self.bopen();

        self.print_inner_attributes(attrs);

        for st in blk.stmts {
            self.print_stmt(st);
        }
        if let Some(ref expr) = blk.expr {
            self.space_if_not_bol();
            self.print_expr(&expr);
            self.maybe_print_trailing_comment(expr.span, Some(blk.span.hi()));
        }
        self.bclose_maybe_open(blk.span, close_box);
        self.ann.post(self, AnnNode::Block(blk))
    }

    fn print_else(&mut self, els: Option<&hir::Expr<'_>>) {
        if let Some(els_inner) = els {
            match els_inner.kind {
                // Another `else if` block.
                hir::ExprKind::If(ref i, ref then, ref e) => {
                    self.cbox(INDENT_UNIT - 1);
                    self.ibox(0);
                    self.s.word(" else if ");
                    self.print_expr_as_cond(&i);
                    self.s.space();
                    self.print_expr(&then);
                    self.print_else(e.as_ref().map(|e| &**e))
                }
                // Final `else` block.
                hir::ExprKind::Block(ref b, _) => {
                    self.cbox(INDENT_UNIT - 1);
                    self.ibox(0);
                    self.s.word(" else ");
                    self.print_block(&b)
                }
                // Constraints would be great here!
                _ => {
                    panic!("print_if saw if with weird alternative");
                }
            }
        }
    }

    pub fn print_if(
        &mut self,
        test: &hir::Expr<'_>,
        blk: &hir::Expr<'_>,
        elseopt: Option<&hir::Expr<'_>>,
    ) {
        self.head("if");
        self.print_expr_as_cond(test);
        self.s.space();
        self.print_expr(blk);
        self.print_else(elseopt)
    }

    pub fn print_anon_const(&mut self, constant: &hir::AnonConst) {
        self.ann.nested(self, Nested::Body(constant.body))
    }

    fn print_call_post(&mut self, args: &[hir::Expr<'_>]) {
        self.popen();
        self.commasep_exprs(Inconsistent, args);
        self.pclose()
    }

    fn print_expr_maybe_paren(&mut self, expr: &hir::Expr<'_>, prec: i8) {
        self.print_expr_cond_paren(expr, expr.precedence().order() < prec)
    }

    /// Prints an expr using syntax that's acceptable in a condition position, such as the `cond` in
    /// `if cond { ... }`.
    pub fn print_expr_as_cond(&mut self, expr: &hir::Expr<'_>) {
        self.print_expr_cond_paren(expr, Self::cond_needs_par(expr))
    }

    /// Prints `expr` or `(expr)` when `needs_par` holds.
    fn print_expr_cond_paren(&mut self, expr: &hir::Expr<'_>, needs_par: bool) {
        if needs_par {
            self.popen();
        }
        if let hir::ExprKind::DropTemps(ref actual_expr) = expr.kind {
            self.print_expr(actual_expr);
        } else {
            self.print_expr(expr);
        }
        if needs_par {
            self.pclose();
        }
    }

    /// Print a `let pat = expr` expression.
    fn print_let(&mut self, pat: &hir::Pat<'_>, expr: &hir::Expr<'_>) {
        self.s.word("let ");
        self.print_pat(pat);
        self.s.space();
        self.word_space("=");
        let npals = || parser::needs_par_as_let_scrutinee(expr.precedence().order());
        self.print_expr_cond_paren(expr, Self::cond_needs_par(expr) || npals())
    }

    // Does `expr` need parenthesis when printed in a condition position?
    //
    // These cases need parens due to the parse error observed in #26461: `if return {}`
    // parses as the erroneous construct `if (return {})`, not `if (return) {}`.
    fn cond_needs_par(expr: &hir::Expr<'_>) -> bool {
        match expr.kind {
            hir::ExprKind::Break(..) | hir::ExprKind::Closure(..) | hir::ExprKind::Ret(..) => true,
            _ => contains_exterior_struct_lit(expr),
        }
    }

    fn print_expr_vec(&mut self, exprs: &[hir::Expr<'_>]) {
        self.ibox(INDENT_UNIT);
        self.s.word("[");
        self.commasep_exprs(Inconsistent, exprs);
        self.s.word("]");
        self.end()
    }

    fn print_expr_anon_const(&mut self, anon_const: &hir::AnonConst) {
        self.ibox(INDENT_UNIT);
        self.s.word_space("const");
        self.print_anon_const(anon_const);
        self.end()
    }

    fn print_expr_repeat(&mut self, element: &hir::Expr<'_>, count: &hir::AnonConst) {
        self.ibox(INDENT_UNIT);
        self.s.word("[");
        self.print_expr(element);
        self.word_space(";");
        self.print_anon_const(count);
        self.s.word("]");
        self.end()
    }

    fn print_expr_struct(
        &mut self,
        qpath: &hir::QPath<'_>,
        fields: &[hir::ExprField<'_>],
        wth: &Option<&hir::Expr<'_>>,
    ) {
        self.print_qpath(qpath, true);
        self.s.word("{");
        self.commasep_cmnt(
            Consistent,
            fields,
            |s, field| {
                s.ibox(INDENT_UNIT);
                if !field.is_shorthand {
                    s.print_ident(field.ident);
                    s.word_space(":");
                }
                s.print_expr(&field.expr);
                s.end()
            },
            |f| f.span,
        );
        match *wth {
            Some(ref expr) => {
                self.ibox(INDENT_UNIT);
                if !fields.is_empty() {
                    self.s.word(",");
                    self.s.space();
                }
                self.s.word("..");
                self.print_expr(&expr);
                self.end();
            }
            _ => {
                if !fields.is_empty() {
                    self.s.word(",")
                }
            }
        }
        self.s.word("}");
    }

    fn print_expr_tup(&mut self, exprs: &[hir::Expr<'_>]) {
        self.popen();
        self.commasep_exprs(Inconsistent, exprs);
        if exprs.len() == 1 {
            self.s.word(",");
        }
        self.pclose()
    }

    fn print_expr_call(&mut self, func: &hir::Expr<'_>, args: &[hir::Expr<'_>]) {
        let prec = match func.kind {
            hir::ExprKind::Field(..) => parser::PREC_FORCE_PAREN,
            _ => parser::PREC_POSTFIX,
        };

        self.print_expr_maybe_paren(func, prec);
        self.print_call_post(args)
    }

    fn print_expr_method_call(&mut self, segment: &hir::PathSegment<'_>, args: &[hir::Expr<'_>]) {
        let base_args = &args[1..];
        self.print_expr_maybe_paren(&args[0], parser::PREC_POSTFIX);
        self.s.word(".");
        self.print_ident(segment.ident);

        let generic_args = segment.args();
        if !generic_args.args.is_empty() || !generic_args.bindings.is_empty() {
            self.print_generic_args(generic_args, segment.infer_args, true);
        }

        self.print_call_post(base_args)
    }

    fn print_expr_binary(&mut self, op: hir::BinOp, lhs: &hir::Expr<'_>, rhs: &hir::Expr<'_>) {
        let assoc_op = bin_op_to_assoc_op(op.node);
        let prec = assoc_op.precedence() as i8;
        let fixity = assoc_op.fixity();

        let (left_prec, right_prec) = match fixity {
            Fixity::Left => (prec, prec + 1),
            Fixity::Right => (prec + 1, prec),
            Fixity::None => (prec + 1, prec + 1),
        };

        let left_prec = match (&lhs.kind, op.node) {
            // These cases need parens: `x as i32 < y` has the parser thinking that `i32 < y` is
            // the beginning of a path type. It starts trying to parse `x as (i32 < y ...` instead
            // of `(x as i32) < ...`. We need to convince it _not_ to do that.
            (&hir::ExprKind::Cast { .. }, hir::BinOpKind::Lt | hir::BinOpKind::Shl) => {
                parser::PREC_FORCE_PAREN
            }
            (&hir::ExprKind::Let { .. }, _) if !parser::needs_par_as_let_scrutinee(prec) => {
                parser::PREC_FORCE_PAREN
            }
            _ => left_prec,
        };

        self.print_expr_maybe_paren(lhs, left_prec);
        self.s.space();
        self.word_space(op.node.as_str());
        self.print_expr_maybe_paren(rhs, right_prec)
    }

    fn print_expr_unary(&mut self, op: hir::UnOp, expr: &hir::Expr<'_>) {
        self.s.word(op.as_str());
        self.print_expr_maybe_paren(expr, parser::PREC_PREFIX)
    }

    fn print_expr_addr_of(
        &mut self,
        kind: hir::BorrowKind,
        mutability: hir::Mutability,
        expr: &hir::Expr<'_>,
    ) {
        self.s.word("&");
        match kind {
            hir::BorrowKind::Ref => self.print_mutability(mutability, false),
            hir::BorrowKind::Raw => {
                self.word_nbsp("raw");
                self.print_mutability(mutability, true);
            }
        }
        self.print_expr_maybe_paren(expr, parser::PREC_PREFIX)
    }

    fn print_literal(&mut self, lit: &hir::Lit) {
        self.maybe_print_comment(lit.span.lo());
        self.word(lit.node.to_lit_token().to_string())
    }

    fn print_inline_asm(&mut self, asm: &hir::InlineAsm<'_>) {
        enum AsmArg<'a> {
            Template(String),
            Operand(&'a hir::InlineAsmOperand<'a>),
            Options(ast::InlineAsmOptions),
        }

        let mut args =
            vec![AsmArg::Template(ast::InlineAsmTemplatePiece::to_string(&asm.template))];
        args.extend(asm.operands.iter().map(|(o, _)| AsmArg::Operand(o)));
        if !asm.options.is_empty() {
            args.push(AsmArg::Options(asm.options));
        }

        self.popen();
        self.commasep(Consistent, &args, |s, arg| match arg {
            AsmArg::Template(template) => s.print_string(&template, ast::StrStyle::Cooked),
            AsmArg::Operand(op) => match op {
                hir::InlineAsmOperand::In { reg, expr } => {
                    s.word("in");
                    s.popen();
                    s.word(format!("{}", reg));
                    s.pclose();
                    s.space();
                    s.print_expr(expr);
                }
                hir::InlineAsmOperand::Out { reg, late, expr } => {
                    s.word(if *late { "lateout" } else { "out" });
                    s.popen();
                    s.word(format!("{}", reg));
                    s.pclose();
                    s.space();
                    match expr {
                        Some(expr) => s.print_expr(expr),
                        None => s.word("_"),
                    }
                }
                hir::InlineAsmOperand::InOut { reg, late, expr } => {
                    s.word(if *late { "inlateout" } else { "inout" });
                    s.popen();
                    s.word(format!("{}", reg));
                    s.pclose();
                    s.space();
                    s.print_expr(expr);
                }
                hir::InlineAsmOperand::SplitInOut { reg, late, in_expr, out_expr } => {
                    s.word(if *late { "inlateout" } else { "inout" });
                    s.popen();
                    s.word(format!("{}", reg));
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
                hir::InlineAsmOperand::Const { anon_const } => {
                    s.word("const");
                    s.space();
                    s.print_anon_const(anon_const);
                }
                hir::InlineAsmOperand::Sym { expr } => {
                    s.word("sym");
                    s.space();
                    s.print_expr(expr);
                }
            },
            AsmArg::Options(opts) => {
                s.word("options");
                s.popen();
                let mut options = vec![];
                if opts.contains(ast::InlineAsmOptions::PURE) {
                    options.push("pure");
                }
                if opts.contains(ast::InlineAsmOptions::NOMEM) {
                    options.push("nomem");
                }
                if opts.contains(ast::InlineAsmOptions::READONLY) {
                    options.push("readonly");
                }
                if opts.contains(ast::InlineAsmOptions::PRESERVES_FLAGS) {
                    options.push("preserves_flags");
                }
                if opts.contains(ast::InlineAsmOptions::NORETURN) {
                    options.push("noreturn");
                }
                if opts.contains(ast::InlineAsmOptions::NOSTACK) {
                    options.push("nostack");
                }
                if opts.contains(ast::InlineAsmOptions::ATT_SYNTAX) {
                    options.push("att_syntax");
                }
                if opts.contains(ast::InlineAsmOptions::RAW) {
                    options.push("raw");
                }
                s.commasep(Inconsistent, &options, |s, &opt| {
                    s.word(opt);
                });
                s.pclose();
            }
        });
        self.pclose();
    }

    pub fn print_expr(&mut self, expr: &hir::Expr<'_>) {
        self.maybe_print_comment(expr.span.lo());
        self.print_outer_attributes(self.attrs(expr.hir_id));
        self.ibox(INDENT_UNIT);
        self.ann.pre(self, AnnNode::Expr(expr));
        match expr.kind {
            hir::ExprKind::Box(ref expr) => {
                self.word_space("box");
                self.print_expr_maybe_paren(expr, parser::PREC_PREFIX);
            }
            hir::ExprKind::Array(ref exprs) => {
                self.print_expr_vec(exprs);
            }
            hir::ExprKind::ConstBlock(ref anon_const) => {
                self.print_expr_anon_const(anon_const);
            }
            hir::ExprKind::Repeat(ref element, ref count) => {
                self.print_expr_repeat(&element, count);
            }
            hir::ExprKind::Struct(ref qpath, fields, ref wth) => {
                self.print_expr_struct(qpath, fields, wth);
            }
            hir::ExprKind::Tup(ref exprs) => {
                self.print_expr_tup(exprs);
            }
            hir::ExprKind::Call(ref func, ref args) => {
                self.print_expr_call(&func, args);
            }
            hir::ExprKind::MethodCall(ref segment, _, ref args, _) => {
                self.print_expr_method_call(segment, args);
            }
            hir::ExprKind::Binary(op, ref lhs, ref rhs) => {
                self.print_expr_binary(op, &lhs, &rhs);
            }
            hir::ExprKind::Unary(op, ref expr) => {
                self.print_expr_unary(op, &expr);
            }
            hir::ExprKind::AddrOf(k, m, ref expr) => {
                self.print_expr_addr_of(k, m, &expr);
            }
            hir::ExprKind::Lit(ref lit) => {
                self.print_literal(&lit);
            }
            hir::ExprKind::Cast(ref expr, ref ty) => {
                let prec = AssocOp::As.precedence() as i8;
                self.print_expr_maybe_paren(&expr, prec);
                self.s.space();
                self.word_space("as");
                self.print_type(&ty);
            }
            hir::ExprKind::Type(ref expr, ref ty) => {
                let prec = AssocOp::Colon.precedence() as i8;
                self.print_expr_maybe_paren(&expr, prec);
                self.word_space(":");
                self.print_type(&ty);
            }
            hir::ExprKind::DropTemps(ref init) => {
                // Print `{`:
                self.cbox(INDENT_UNIT);
                self.ibox(0);
                self.bopen();

                // Print `let _t = $init;`:
                let temp = Ident::from_str("_t");
                self.print_local(Some(init), |this| this.print_ident(temp));
                self.s.word(";");

                // Print `_t`:
                self.space_if_not_bol();
                self.print_ident(temp);

                // Print `}`:
                self.bclose_maybe_open(expr.span, true);
            }
            hir::ExprKind::Let(ref pat, ref scrutinee, _) => {
                self.print_let(pat, scrutinee);
            }
            hir::ExprKind::If(ref test, ref blk, ref elseopt) => {
                self.print_if(&test, &blk, elseopt.as_ref().map(|e| &**e));
            }
            hir::ExprKind::Loop(ref blk, opt_label, _, _) => {
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }
                self.head("loop");
                self.print_block(&blk);
            }
            hir::ExprKind::Match(ref expr, arms, _) => {
                self.cbox(INDENT_UNIT);
                self.ibox(INDENT_UNIT);
                self.word_nbsp("match");
                self.print_expr_as_cond(&expr);
                self.s.space();
                self.bopen();
                for arm in arms {
                    self.print_arm(arm);
                }
                self.bclose(expr.span);
            }
            hir::ExprKind::Closure(capture_clause, ref decl, body, _fn_decl_span, _gen) => {
                self.print_capture_clause(capture_clause);

                self.print_closure_params(&decl, body);
                self.s.space();

                // This is a bare expression.
                self.ann.nested(self, Nested::Body(body));
                self.end(); // need to close a box

                // A box will be closed by `print_expr`, but we didn't want an overall
                // wrapper so we closed the corresponding opening. so create an
                // empty box to satisfy the close.
                self.ibox(0);
            }
            hir::ExprKind::Block(ref blk, opt_label) => {
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }
                // containing cbox, will be closed by print-block at `}`
                self.cbox(INDENT_UNIT);
                // head-box, will be closed by print-block after `{`
                self.ibox(0);
                self.print_block(&blk);
            }
            hir::ExprKind::Assign(ref lhs, ref rhs, _) => {
                let prec = AssocOp::Assign.precedence() as i8;
                self.print_expr_maybe_paren(&lhs, prec + 1);
                self.s.space();
                self.word_space("=");
                self.print_expr_maybe_paren(&rhs, prec);
            }
            hir::ExprKind::AssignOp(op, ref lhs, ref rhs) => {
                let prec = AssocOp::Assign.precedence() as i8;
                self.print_expr_maybe_paren(&lhs, prec + 1);
                self.s.space();
                self.s.word(op.node.as_str());
                self.word_space("=");
                self.print_expr_maybe_paren(&rhs, prec);
            }
            hir::ExprKind::Field(ref expr, ident) => {
                self.print_expr_maybe_paren(expr, parser::PREC_POSTFIX);
                self.s.word(".");
                self.print_ident(ident);
            }
            hir::ExprKind::Index(ref expr, ref index) => {
                self.print_expr_maybe_paren(&expr, parser::PREC_POSTFIX);
                self.s.word("[");
                self.print_expr(&index);
                self.s.word("]");
            }
            hir::ExprKind::Path(ref qpath) => self.print_qpath(qpath, true),
            hir::ExprKind::Break(destination, ref opt_expr) => {
                self.s.word("break");
                self.s.space();
                if let Some(label) = destination.label {
                    self.print_ident(label.ident);
                    self.s.space();
                }
                if let Some(ref expr) = *opt_expr {
                    self.print_expr_maybe_paren(expr, parser::PREC_JUMP);
                    self.s.space();
                }
            }
            hir::ExprKind::Continue(destination) => {
                self.s.word("continue");
                self.s.space();
                if let Some(label) = destination.label {
                    self.print_ident(label.ident);
                    self.s.space()
                }
            }
            hir::ExprKind::Ret(ref result) => {
                self.s.word("return");
                if let Some(ref expr) = *result {
                    self.s.word(" ");
                    self.print_expr_maybe_paren(&expr, parser::PREC_JUMP);
                }
            }
            hir::ExprKind::InlineAsm(ref asm) => {
                self.word("asm!");
                self.print_inline_asm(asm);
            }
            hir::ExprKind::LlvmInlineAsm(ref a) => {
                let i = &a.inner;
                self.s.word("llvm_asm!");
                self.popen();
                self.print_symbol(i.asm, i.asm_str_style);
                self.word_space(":");

                let mut out_idx = 0;
                self.commasep(Inconsistent, &i.outputs, |s, out| {
                    let constraint = out.constraint.as_str();
                    let mut ch = constraint.chars();
                    match ch.next() {
                        Some('=') if out.is_rw => {
                            s.print_string(&format!("+{}", ch.as_str()), ast::StrStyle::Cooked)
                        }
                        _ => s.print_string(&constraint, ast::StrStyle::Cooked),
                    }
                    s.popen();
                    s.print_expr(&a.outputs_exprs[out_idx]);
                    s.pclose();
                    out_idx += 1;
                });
                self.s.space();
                self.word_space(":");

                let mut in_idx = 0;
                self.commasep(Inconsistent, &i.inputs, |s, &co| {
                    s.print_symbol(co, ast::StrStyle::Cooked);
                    s.popen();
                    s.print_expr(&a.inputs_exprs[in_idx]);
                    s.pclose();
                    in_idx += 1;
                });
                self.s.space();
                self.word_space(":");

                self.commasep(Inconsistent, &i.clobbers, |s, &co| {
                    s.print_symbol(co, ast::StrStyle::Cooked);
                });

                let mut options = vec![];
                if i.volatile {
                    options.push("volatile");
                }
                if i.alignstack {
                    options.push("alignstack");
                }
                if i.dialect == ast::LlvmAsmDialect::Intel {
                    options.push("intel");
                }

                if !options.is_empty() {
                    self.s.space();
                    self.word_space(":");
                    self.commasep(Inconsistent, &options, |s, &co| {
                        s.print_string(co, ast::StrStyle::Cooked);
                    });
                }

                self.pclose();
            }
            hir::ExprKind::Yield(ref expr, _) => {
                self.word_space("yield");
                self.print_expr_maybe_paren(&expr, parser::PREC_JUMP);
            }
            hir::ExprKind::Err => {
                self.popen();
                self.s.word("/*ERROR*/");
                self.pclose();
            }
        }
        self.ann.post(self, AnnNode::Expr(expr));
        self.end()
    }

    pub fn print_local_decl(&mut self, loc: &hir::Local<'_>) {
        self.print_pat(&loc.pat);
        if let Some(ref ty) = loc.ty {
            self.word_space(":");
            self.print_type(&ty);
        }
    }

    pub fn print_name(&mut self, name: Symbol) {
        self.print_ident(Ident::with_dummy_span(name))
    }

    pub fn print_path(&mut self, path: &hir::Path<'_>, colons_before_params: bool) {
        self.maybe_print_comment(path.span.lo());

        for (i, segment) in path.segments.iter().enumerate() {
            if i > 0 {
                self.s.word("::")
            }
            if segment.ident.name != kw::PathRoot {
                self.print_ident(segment.ident);
                self.print_generic_args(segment.args(), segment.infer_args, colons_before_params);
            }
        }
    }

    pub fn print_path_segment(&mut self, segment: &hir::PathSegment<'_>) {
        if segment.ident.name != kw::PathRoot {
            self.print_ident(segment.ident);
            self.print_generic_args(segment.args(), segment.infer_args, false);
        }
    }

    pub fn print_qpath(&mut self, qpath: &hir::QPath<'_>, colons_before_params: bool) {
        match *qpath {
            hir::QPath::Resolved(None, ref path) => self.print_path(path, colons_before_params),
            hir::QPath::Resolved(Some(ref qself), ref path) => {
                self.s.word("<");
                self.print_type(qself);
                self.s.space();
                self.word_space("as");

                for (i, segment) in path.segments[..path.segments.len() - 1].iter().enumerate() {
                    if i > 0 {
                        self.s.word("::")
                    }
                    if segment.ident.name != kw::PathRoot {
                        self.print_ident(segment.ident);
                        self.print_generic_args(
                            segment.args(),
                            segment.infer_args,
                            colons_before_params,
                        );
                    }
                }

                self.s.word(">");
                self.s.word("::");
                let item_segment = path.segments.last().unwrap();
                self.print_ident(item_segment.ident);
                self.print_generic_args(
                    item_segment.args(),
                    item_segment.infer_args,
                    colons_before_params,
                )
            }
            hir::QPath::TypeRelative(ref qself, ref item_segment) => {
                // If we've got a compound-qualified-path, let's push an additional pair of angle
                // brackets, so that we pretty-print `<<A::B>::C>` as `<A::B>::C`, instead of just
                // `A::B::C` (since the latter could be ambiguous to the user)
                if let hir::TyKind::Path(hir::QPath::Resolved(None, _)) = &qself.kind {
                    self.print_type(qself);
                } else {
                    self.s.word("<");
                    self.print_type(qself);
                    self.s.word(">");
                }

                self.s.word("::");
                self.print_ident(item_segment.ident);
                self.print_generic_args(
                    item_segment.args(),
                    item_segment.infer_args,
                    colons_before_params,
                )
            }
            hir::QPath::LangItem(lang_item, span) => {
                self.s.word("#[lang = \"");
                self.print_ident(Ident::new(lang_item.name(), span));
                self.s.word("\"]");
            }
        }
    }

    fn print_generic_args(
        &mut self,
        generic_args: &hir::GenericArgs<'_>,
        infer_args: bool,
        colons_before_params: bool,
    ) {
        if generic_args.parenthesized {
            self.s.word("(");
            self.commasep(Inconsistent, generic_args.inputs(), |s, ty| s.print_type(&ty));
            self.s.word(")");

            self.space_if_not_bol();
            self.word_space("->");
            self.print_type(generic_args.bindings[0].ty());
        } else {
            let start = if colons_before_params { "::<" } else { "<" };
            let empty = Cell::new(true);
            let start_or_comma = |this: &mut Self| {
                if empty.get() {
                    empty.set(false);
                    this.s.word(start)
                } else {
                    this.word_space(",")
                }
            };

            let mut nonelided_generic_args: bool = false;
            let elide_lifetimes = generic_args.args.iter().all(|arg| match arg {
                GenericArg::Lifetime(lt) => lt.is_elided(),
                _ => {
                    nonelided_generic_args = true;
                    true
                }
            });

            if nonelided_generic_args {
                start_or_comma(self);
                self.commasep(
                    Inconsistent,
                    &generic_args.args,
                    |s, generic_arg| match generic_arg {
                        GenericArg::Lifetime(lt) if !elide_lifetimes => s.print_lifetime(lt),
                        GenericArg::Lifetime(_) => {}
                        GenericArg::Type(ty) => s.print_type(ty),
                        GenericArg::Const(ct) => s.print_anon_const(&ct.value),
                        GenericArg::Infer(_inf) => s.word("_"),
                    },
                );
            }

            // FIXME(eddyb): this would leak into error messages (e.g.,
            // "non-exhaustive patterns: `Some::<..>(_)` not covered").
            if infer_args && false {
                start_or_comma(self);
                self.s.word("..");
            }

            for binding in generic_args.bindings.iter() {
                start_or_comma(self);
                self.print_ident(binding.ident);
                self.print_generic_args(binding.gen_args, false, false);
                self.s.space();
                match generic_args.bindings[0].kind {
                    hir::TypeBindingKind::Equality { ref ty } => {
                        self.word_space("=");
                        self.print_type(ty);
                    }
                    hir::TypeBindingKind::Constraint { bounds } => {
                        self.print_bounds(":", bounds);
                    }
                }
            }

            if !empty.get() {
                self.s.word(">")
            }
        }
    }

    pub fn print_pat(&mut self, pat: &hir::Pat<'_>) {
        self.maybe_print_comment(pat.span.lo());
        self.ann.pre(self, AnnNode::Pat(pat));
        // Pat isn't normalized, but the beauty of it
        // is that it doesn't matter
        match pat.kind {
            PatKind::Wild => self.s.word("_"),
            PatKind::Binding(binding_mode, _, ident, ref sub) => {
                match binding_mode {
                    hir::BindingAnnotation::Ref => {
                        self.word_nbsp("ref");
                        self.print_mutability(hir::Mutability::Not, false);
                    }
                    hir::BindingAnnotation::RefMut => {
                        self.word_nbsp("ref");
                        self.print_mutability(hir::Mutability::Mut, false);
                    }
                    hir::BindingAnnotation::Unannotated => {}
                    hir::BindingAnnotation::Mutable => {
                        self.word_nbsp("mut");
                    }
                }
                self.print_ident(ident);
                if let Some(ref p) = *sub {
                    self.s.word("@");
                    self.print_pat(&p);
                }
            }
            PatKind::TupleStruct(ref qpath, ref elts, ddpos) => {
                self.print_qpath(qpath, true);
                self.popen();
                if let Some(ddpos) = ddpos {
                    self.commasep(Inconsistent, &elts[..ddpos], |s, p| s.print_pat(&p));
                    if ddpos != 0 {
                        self.word_space(",");
                    }
                    self.s.word("..");
                    if ddpos != elts.len() {
                        self.s.word(",");
                        self.commasep(Inconsistent, &elts[ddpos..], |s, p| s.print_pat(&p));
                    }
                } else {
                    self.commasep(Inconsistent, &elts[..], |s, p| s.print_pat(&p));
                }
                self.pclose();
            }
            PatKind::Path(ref qpath) => {
                self.print_qpath(qpath, true);
            }
            PatKind::Struct(ref qpath, ref fields, etc) => {
                self.print_qpath(qpath, true);
                self.nbsp();
                self.word_space("{");
                self.commasep_cmnt(
                    Consistent,
                    &fields[..],
                    |s, f| {
                        s.cbox(INDENT_UNIT);
                        if !f.is_shorthand {
                            s.print_ident(f.ident);
                            s.word_nbsp(":");
                        }
                        s.print_pat(&f.pat);
                        s.end()
                    },
                    |f| f.pat.span,
                );
                if etc {
                    if !fields.is_empty() {
                        self.word_space(",");
                    }
                    self.s.word("..");
                }
                self.s.space();
                self.s.word("}");
            }
            PatKind::Or(ref pats) => {
                self.strsep("|", true, Inconsistent, &pats[..], |s, p| s.print_pat(&p));
            }
            PatKind::Tuple(ref elts, ddpos) => {
                self.popen();
                if let Some(ddpos) = ddpos {
                    self.commasep(Inconsistent, &elts[..ddpos], |s, p| s.print_pat(&p));
                    if ddpos != 0 {
                        self.word_space(",");
                    }
                    self.s.word("..");
                    if ddpos != elts.len() {
                        self.s.word(",");
                        self.commasep(Inconsistent, &elts[ddpos..], |s, p| s.print_pat(&p));
                    }
                } else {
                    self.commasep(Inconsistent, &elts[..], |s, p| s.print_pat(&p));
                    if elts.len() == 1 {
                        self.s.word(",");
                    }
                }
                self.pclose();
            }
            PatKind::Box(ref inner) => {
                let is_range_inner = matches!(inner.kind, PatKind::Range(..));
                self.s.word("box ");
                if is_range_inner {
                    self.popen();
                }
                self.print_pat(&inner);
                if is_range_inner {
                    self.pclose();
                }
            }
            PatKind::Ref(ref inner, mutbl) => {
                let is_range_inner = matches!(inner.kind, PatKind::Range(..));
                self.s.word("&");
                self.s.word(mutbl.prefix_str());
                if is_range_inner {
                    self.popen();
                }
                self.print_pat(&inner);
                if is_range_inner {
                    self.pclose();
                }
            }
            PatKind::Lit(ref e) => self.print_expr(&e),
            PatKind::Range(ref begin, ref end, ref end_kind) => {
                if let Some(expr) = begin {
                    self.print_expr(expr);
                    self.s.space();
                }
                match *end_kind {
                    RangeEnd::Included => self.s.word("..."),
                    RangeEnd::Excluded => self.s.word(".."),
                }
                if let Some(expr) = end {
                    self.print_expr(expr);
                }
            }
            PatKind::Slice(ref before, ref slice, ref after) => {
                self.s.word("[");
                self.commasep(Inconsistent, &before[..], |s, p| s.print_pat(&p));
                if let Some(ref p) = *slice {
                    if !before.is_empty() {
                        self.word_space(",");
                    }
                    if let PatKind::Wild = p.kind {
                        // Print nothing.
                    } else {
                        self.print_pat(&p);
                    }
                    self.s.word("..");
                    if !after.is_empty() {
                        self.word_space(",");
                    }
                }
                self.commasep(Inconsistent, &after[..], |s, p| s.print_pat(&p));
                self.s.word("]");
            }
        }
        self.ann.post(self, AnnNode::Pat(pat))
    }

    pub fn print_param(&mut self, arg: &hir::Param<'_>) {
        self.print_outer_attributes(self.attrs(arg.hir_id));
        self.print_pat(&arg.pat);
    }

    pub fn print_arm(&mut self, arm: &hir::Arm<'_>) {
        // I have no idea why this check is necessary, but here it
        // is :(
        if self.attrs(arm.hir_id).is_empty() {
            self.s.space();
        }
        self.cbox(INDENT_UNIT);
        self.ann.pre(self, AnnNode::Arm(arm));
        self.ibox(0);
        self.print_outer_attributes(&self.attrs(arm.hir_id));
        self.print_pat(&arm.pat);
        self.s.space();
        if let Some(ref g) = arm.guard {
            match g {
                hir::Guard::If(e) => {
                    self.word_space("if");
                    self.print_expr(&e);
                    self.s.space();
                }
                hir::Guard::IfLet(pat, e) => {
                    self.word_nbsp("if");
                    self.word_nbsp("let");
                    self.print_pat(&pat);
                    self.s.space();
                    self.word_space("=");
                    self.print_expr(&e);
                    self.s.space();
                }
            }
        }
        self.word_space("=>");

        match arm.body.kind {
            hir::ExprKind::Block(ref blk, opt_label) => {
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }
                // the block will close the pattern's ibox
                self.print_block_unclosed(&blk);

                // If it is a user-provided unsafe block, print a comma after it
                if let hir::BlockCheckMode::UnsafeBlock(hir::UnsafeSource::UserProvided) = blk.rules
                {
                    self.s.word(",");
                }
            }
            _ => {
                self.end(); // close the ibox for the pattern
                self.print_expr(&arm.body);
                self.s.word(",");
            }
        }
        self.ann.post(self, AnnNode::Arm(arm));
        self.end() // close enclosing cbox
    }

    pub fn print_fn(
        &mut self,
        decl: &hir::FnDecl<'_>,
        header: hir::FnHeader,
        name: Option<Symbol>,
        generics: &hir::Generics<'_>,
        vis: &hir::Visibility<'_>,
        arg_names: &[Ident],
        body_id: Option<hir::BodyId>,
    ) {
        self.print_fn_header_info(header, vis);

        if let Some(name) = name {
            self.nbsp();
            self.print_name(name);
        }
        self.print_generic_params(&generics.params);

        self.popen();
        let mut i = 0;
        // Make sure we aren't supplied *both* `arg_names` and `body_id`.
        assert!(arg_names.is_empty() || body_id.is_none());
        self.commasep(Inconsistent, &decl.inputs, |s, ty| {
            s.ibox(INDENT_UNIT);
            if let Some(arg_name) = arg_names.get(i) {
                s.s.word(arg_name.to_string());
                s.s.word(":");
                s.s.space();
            } else if let Some(body_id) = body_id {
                s.ann.nested(s, Nested::BodyParamPat(body_id, i));
                s.s.word(":");
                s.s.space();
            }
            i += 1;
            s.print_type(ty);
            s.end()
        });
        if decl.c_variadic {
            self.s.word(", ...");
        }
        self.pclose();

        self.print_fn_output(decl);
        self.print_where_clause(&generics.where_clause)
    }

    fn print_closure_params(&mut self, decl: &hir::FnDecl<'_>, body_id: hir::BodyId) {
        self.s.word("|");
        let mut i = 0;
        self.commasep(Inconsistent, &decl.inputs, |s, ty| {
            s.ibox(INDENT_UNIT);

            s.ann.nested(s, Nested::BodyParamPat(body_id, i));
            i += 1;

            if let hir::TyKind::Infer = ty.kind {
                // Print nothing.
            } else {
                s.s.word(":");
                s.s.space();
                s.print_type(ty);
            }
            s.end();
        });
        self.s.word("|");

        if let hir::FnRetTy::DefaultReturn(..) = decl.output {
            return;
        }

        self.space_if_not_bol();
        self.word_space("->");
        match decl.output {
            hir::FnRetTy::Return(ref ty) => {
                self.print_type(&ty);
                self.maybe_print_comment(ty.span.lo())
            }
            hir::FnRetTy::DefaultReturn(..) => unreachable!(),
        }
    }

    pub fn print_capture_clause(&mut self, capture_clause: hir::CaptureBy) {
        match capture_clause {
            hir::CaptureBy::Value => self.word_space("move"),
            hir::CaptureBy::Ref => {}
        }
    }

    pub fn print_bounds<'b>(
        &mut self,
        prefix: &'static str,
        bounds: impl IntoIterator<Item = &'b hir::GenericBound<'b>>,
    ) {
        let mut first = true;
        for bound in bounds {
            if first {
                self.s.word(prefix);
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
                GenericBound::Trait(tref, modifier) => {
                    if modifier == &TraitBoundModifier::Maybe {
                        self.s.word("?");
                    }
                    self.print_poly_trait_ref(tref);
                }
                GenericBound::LangItemTrait(lang_item, span, ..) => {
                    self.s.word("#[lang = \"");
                    self.print_ident(Ident::new(lang_item.name(), *span));
                    self.s.word("\"]");
                }
                GenericBound::Outlives(lt) => {
                    self.print_lifetime(lt);
                }
            }
        }
    }

    pub fn print_generic_params(&mut self, generic_params: &[GenericParam<'_>]) {
        if !generic_params.is_empty() {
            self.s.word("<");

            self.commasep(Inconsistent, generic_params, |s, param| s.print_generic_param(param));

            self.s.word(">");
        }
    }

    pub fn print_generic_param(&mut self, param: &GenericParam<'_>) {
        if let GenericParamKind::Const { .. } = param.kind {
            self.word_space("const");
        }

        self.print_ident(param.name.ident());

        match param.kind {
            GenericParamKind::Lifetime { .. } => {
                let mut sep = ":";
                for bound in param.bounds {
                    match bound {
                        GenericBound::Outlives(ref lt) => {
                            self.s.word(sep);
                            self.print_lifetime(lt);
                            sep = "+";
                        }
                        _ => panic!(),
                    }
                }
            }
            GenericParamKind::Type { ref default, .. } => {
                self.print_bounds(":", param.bounds);
                if let Some(default) = default {
                    self.s.space();
                    self.word_space("=");
                    self.print_type(&default)
                }
            }
            GenericParamKind::Const { ref ty, ref default } => {
                self.word_space(":");
                self.print_type(ty);
                if let Some(ref default) = default {
                    self.s.space();
                    self.word_space("=");
                    self.print_anon_const(&default)
                }
            }
        }
    }

    pub fn print_lifetime(&mut self, lifetime: &hir::Lifetime) {
        self.print_ident(lifetime.name.ident())
    }

    pub fn print_where_clause(&mut self, where_clause: &hir::WhereClause<'_>) {
        if where_clause.predicates.is_empty() {
            return;
        }

        self.s.space();
        self.word_space("where");

        for (i, predicate) in where_clause.predicates.iter().enumerate() {
            if i != 0 {
                self.word_space(",");
            }

            match predicate {
                hir::WherePredicate::BoundPredicate(hir::WhereBoundPredicate {
                    bound_generic_params,
                    bounded_ty,
                    bounds,
                    ..
                }) => {
                    self.print_formal_generic_params(bound_generic_params);
                    self.print_type(&bounded_ty);
                    self.print_bounds(":", *bounds);
                }
                hir::WherePredicate::RegionPredicate(hir::WhereRegionPredicate {
                    lifetime,
                    bounds,
                    ..
                }) => {
                    self.print_lifetime(lifetime);
                    self.s.word(":");

                    for (i, bound) in bounds.iter().enumerate() {
                        match bound {
                            GenericBound::Outlives(lt) => {
                                self.print_lifetime(lt);
                            }
                            _ => panic!(),
                        }

                        if i != 0 {
                            self.s.word(":");
                        }
                    }
                }
                hir::WherePredicate::EqPredicate(hir::WhereEqPredicate {
                    lhs_ty, rhs_ty, ..
                }) => {
                    self.print_type(lhs_ty);
                    self.s.space();
                    self.word_space("=");
                    self.print_type(rhs_ty);
                }
            }
        }
    }

    pub fn print_mutability(&mut self, mutbl: hir::Mutability, print_const: bool) {
        match mutbl {
            hir::Mutability::Mut => self.word_nbsp("mut"),
            hir::Mutability::Not => {
                if print_const {
                    self.word_nbsp("const")
                }
            }
        }
    }

    pub fn print_mt(&mut self, mt: &hir::MutTy<'_>, print_const: bool) {
        self.print_mutability(mt.mutbl, print_const);
        self.print_type(&mt.ty)
    }

    pub fn print_fn_output(&mut self, decl: &hir::FnDecl<'_>) {
        if let hir::FnRetTy::DefaultReturn(..) = decl.output {
            return;
        }

        self.space_if_not_bol();
        self.ibox(INDENT_UNIT);
        self.word_space("->");
        match decl.output {
            hir::FnRetTy::DefaultReturn(..) => unreachable!(),
            hir::FnRetTy::Return(ref ty) => self.print_type(&ty),
        }
        self.end();

        if let hir::FnRetTy::Return(ref output) = decl.output {
            self.maybe_print_comment(output.span.lo())
        }
    }

    pub fn print_ty_fn(
        &mut self,
        abi: Abi,
        unsafety: hir::Unsafety,
        decl: &hir::FnDecl<'_>,
        name: Option<Symbol>,
        generic_params: &[hir::GenericParam<'_>],
        arg_names: &[Ident],
    ) {
        self.ibox(INDENT_UNIT);
        if !generic_params.is_empty() {
            self.s.word("for");
            self.print_generic_params(generic_params);
        }
        let generics = hir::Generics {
            params: &[],
            where_clause: hir::WhereClause { predicates: &[], span: rustc_span::DUMMY_SP },
            span: rustc_span::DUMMY_SP,
        };
        self.print_fn(
            decl,
            hir::FnHeader {
                unsafety,
                abi,
                constness: hir::Constness::NotConst,
                asyncness: hir::IsAsync::NotAsync,
            },
            name,
            &generics,
            &Spanned { span: rustc_span::DUMMY_SP, node: hir::VisibilityKind::Inherited },
            arg_names,
            None,
        );
        self.end();
    }

    pub fn maybe_print_trailing_comment(
        &mut self,
        span: rustc_span::Span,
        next_pos: Option<BytePos>,
    ) {
        if let Some(cmnts) = self.comments() {
            if let Some(cmnt) = cmnts.trailing_comment(span, next_pos) {
                self.print_comment(&cmnt);
            }
        }
    }

    pub fn print_remaining_comments(&mut self) {
        // If there aren't any remaining comments, then we need to manually
        // make sure there is a line break at the end.
        if self.next_comment().is_none() {
            self.s.hardbreak();
        }
        while let Some(ref cmnt) = self.next_comment() {
            self.print_comment(cmnt)
        }
    }

    pub fn print_fn_header_info(&mut self, header: hir::FnHeader, vis: &hir::Visibility<'_>) {
        self.s.word(visibility_qualified(vis, ""));

        match header.constness {
            hir::Constness::NotConst => {}
            hir::Constness::Const => self.word_nbsp("const"),
        }

        match header.asyncness {
            hir::IsAsync::NotAsync => {}
            hir::IsAsync::Async => self.word_nbsp("async"),
        }

        self.print_unsafety(header.unsafety);

        if header.abi != Abi::Rust {
            self.word_nbsp("extern");
            self.word_nbsp(header.abi.to_string());
        }

        self.s.word("fn")
    }

    pub fn print_unsafety(&mut self, s: hir::Unsafety) {
        match s {
            hir::Unsafety::Normal => {}
            hir::Unsafety::Unsafe => self.word_nbsp("unsafe"),
        }
    }

    pub fn print_is_auto(&mut self, s: hir::IsAuto) {
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
        hir::StmtKind::Local(_) => true,
        hir::StmtKind::Item(_) => false,
        hir::StmtKind::Expr(ref e) => expr_requires_semi_to_be_stmt(&e),
        hir::StmtKind::Semi(..) => false,
    }
}

fn bin_op_to_assoc_op(op: hir::BinOpKind) -> AssocOp {
    use crate::hir::BinOpKind::*;
    match op {
        Add => AssocOp::Add,
        Sub => AssocOp::Subtract,
        Mul => AssocOp::Multiply,
        Div => AssocOp::Divide,
        Rem => AssocOp::Modulus,

        And => AssocOp::LAnd,
        Or => AssocOp::LOr,

        BitXor => AssocOp::BitXor,
        BitAnd => AssocOp::BitAnd,
        BitOr => AssocOp::BitOr,
        Shl => AssocOp::ShiftLeft,
        Shr => AssocOp::ShiftRight,

        Eq => AssocOp::Equal,
        Lt => AssocOp::Less,
        Le => AssocOp::LessEqual,
        Ne => AssocOp::NotEqual,
        Ge => AssocOp::GreaterEqual,
        Gt => AssocOp::Greater,
    }
}

/// Expressions that syntactically contain an "exterior" struct literal, i.e., not surrounded by any
/// parens or other delimiters, e.g., `X { y: 1 }`, `X { y: 1 }.method()`, `foo == X { y: 1 }` and
/// `X { y: 1 } == foo` all do, but `(X { y: 1 }) == foo` does not.
fn contains_exterior_struct_lit(value: &hir::Expr<'_>) -> bool {
    match value.kind {
        hir::ExprKind::Struct(..) => true,

        hir::ExprKind::Assign(ref lhs, ref rhs, _)
        | hir::ExprKind::AssignOp(_, ref lhs, ref rhs)
        | hir::ExprKind::Binary(_, ref lhs, ref rhs) => {
            // `X { y: 1 } + X { y: 2 }`
            contains_exterior_struct_lit(&lhs) || contains_exterior_struct_lit(&rhs)
        }
        hir::ExprKind::Unary(_, ref x)
        | hir::ExprKind::Cast(ref x, _)
        | hir::ExprKind::Type(ref x, _)
        | hir::ExprKind::Field(ref x, _)
        | hir::ExprKind::Index(ref x, _) => {
            // `&X { y: 1 }, X { y: 1 }.y`
            contains_exterior_struct_lit(&x)
        }

        hir::ExprKind::MethodCall(.., ref exprs, _) => {
            // `X { y: 1 }.bar(...)`
            contains_exterior_struct_lit(&exprs[0])
        }

        _ => false,
    }
}
