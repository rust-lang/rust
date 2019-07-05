use rustc_target::spec::abi::Abi;
use syntax::ast;
use syntax::source_map::{SourceMap, Spanned};
use syntax::parse::ParseSess;
use syntax::parse::lexer::comments;
use syntax::print::pp::{self, Breaks};
use syntax::print::pp::Breaks::{Consistent, Inconsistent};
use syntax::print::pprust::{self, PrintState};
use syntax::symbol::kw;
use syntax::util::parser::{self, AssocOp, Fixity};
use syntax_pos::{self, BytePos, FileName};

use crate::hir;
use crate::hir::{PatKind, GenericBound, TraitBoundModifier, RangeEnd};
use crate::hir::{GenericParam, GenericParamKind, GenericArg};
use crate::hir::ptr::P;

use std::borrow::Cow;
use std::cell::Cell;
use std::io::Read;
use std::vec;

pub enum AnnNode<'a> {
    Name(&'a ast::Name),
    Block(&'a hir::Block),
    Item(&'a hir::Item),
    SubItem(hir::HirId),
    Expr(&'a hir::Expr),
    Pat(&'a hir::Pat),
}

pub enum Nested {
    Item(hir::ItemId),
    TraitItem(hir::TraitItemId),
    ImplItem(hir::ImplItemId),
    Body(hir::BodyId),
    BodyArgPat(hir::BodyId, usize)
}

pub trait PpAnn {
    fn nested(&self, _state: &mut State<'_>, _nested: Nested) {
    }
    fn pre(&self, _state: &mut State<'_>, _node: AnnNode<'_>) {
    }
    fn post(&self, _state: &mut State<'_>, _node: AnnNode<'_>) {
    }
    fn try_fetch_item(&self, _: hir::HirId) -> Option<&hir::Item> {
        None
    }
}

pub struct NoAnn;
impl PpAnn for NoAnn {}
pub const NO_ANN: &dyn PpAnn = &NoAnn;

impl PpAnn for hir::Crate {
    fn try_fetch_item(&self, item: hir::HirId) -> Option<&hir::Item> {
        Some(self.item(item))
    }
    fn nested(&self, state: &mut State<'_>, nested: Nested) {
        match nested {
            Nested::Item(id) => state.print_item(self.item(id.id)),
            Nested::TraitItem(id) => state.print_trait_item(self.trait_item(id)),
            Nested::ImplItem(id) => state.print_impl_item(self.impl_item(id)),
            Nested::Body(id) => state.print_expr(&self.body(id).value),
            Nested::BodyArgPat(id, i) => state.print_pat(&self.body(id).arguments[i].pat)
        }
    }
}

pub struct State<'a> {
    pub s: pp::Printer<'a>,
    cm: Option<&'a SourceMap>,
    comments: Option<Vec<comments::Comment>>,
    cur_cmnt: usize,
    boxes: Vec<pp::Breaks>,
    ann: &'a (dyn PpAnn + 'a),
}

impl<'a> PrintState<'a> for State<'a> {
    fn writer(&mut self) -> &mut pp::Printer<'a> {
        &mut self.s
    }

    fn boxes(&mut self) -> &mut Vec<pp::Breaks> {
        &mut self.boxes
    }

    fn comments(&mut self) -> &mut Option<Vec<comments::Comment>> {
        &mut self.comments
    }

    fn cur_cmnt(&mut self) -> &mut usize {
        &mut self.cur_cmnt
    }
}

#[allow(non_upper_case_globals)]
pub const indent_unit: usize = 4;

/// Requires you to pass an input filename and reader so that
/// it can scan the input text for comments to copy forward.
pub fn print_crate<'a>(cm: &'a SourceMap,
                       sess: &ParseSess,
                       krate: &hir::Crate,
                       filename: FileName,
                       input: &mut dyn Read,
                       out: &'a mut String,
                       ann: &'a dyn PpAnn)
                       {
    let mut s = State::new_from_input(cm, sess, filename, input, out, ann);

    // When printing the AST, we sometimes need to inject `#[no_std]` here.
    // Since you can't compile the HIR, it's not necessary.

    s.print_mod(&krate.module, &krate.attrs);
    s.print_remaining_comments();
    s.s.eof()
}

impl<'a> State<'a> {
    pub fn new_from_input(cm: &'a SourceMap,
                          sess: &ParseSess,
                          filename: FileName,
                          input: &mut dyn Read,
                          out: &'a mut String,
                          ann: &'a dyn PpAnn)
                          -> State<'a> {
        let comments = comments::gather_comments(sess, filename, input);
        State::new(cm, out, ann, Some(comments))
    }

    pub fn new(cm: &'a SourceMap,
               out: &'a mut String,
               ann: &'a dyn PpAnn,
               comments: Option<Vec<comments::Comment>>)
               -> State<'a> {
        State {
            s: pp::mk_printer(out),
            cm: Some(cm),
            comments,
            cur_cmnt: 0,
            boxes: Vec::new(),
            ann,
        }
    }
}

pub fn to_string<F>(ann: &dyn PpAnn, f: F) -> String
    where F: FnOnce(&mut State<'_>)
{
    let mut wr = String::new();
    {
        let mut printer = State {
            s: pp::mk_printer(&mut wr),
            cm: None,
            comments: None,
            cur_cmnt: 0,
            boxes: Vec::new(),
            ann,
        };
        f(&mut printer);
        printer.s.eof();
    }
    wr
}

pub fn visibility_qualified<S: Into<Cow<'static, str>>>(vis: &hir::Visibility, w: S) -> String {
    to_string(NO_ANN, |s| {
        s.print_visibility(vis);
        s.s.word(w)
    })
}

impl<'a> State<'a> {
    pub fn cbox(&mut self, u: usize) {
        self.boxes.push(pp::Breaks::Consistent);
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
        self.cbox(indent_unit);
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

    pub fn bclose_(&mut self, span: syntax_pos::Span, indented: usize) {
        self.bclose_maybe_open(span, indented, true)
    }

    pub fn bclose_maybe_open(&mut self,
                             span: syntax_pos::Span,
                             indented: usize,
                             close_box: bool)
                             {
        self.maybe_print_comment(span.hi());
        self.break_offset_if_not_bol(1, -(indented as isize));
        self.s.word("}");
        if close_box {
            self.end(); // close the outer-box
        }
    }

    pub fn bclose(&mut self, span: syntax_pos::Span) {
        self.bclose_(span, indent_unit)
    }

    pub fn in_cbox(&self) -> bool {
        match self.boxes.last() {
            Some(&last_box) => last_box == pp::Breaks::Consistent,
            None => false,
        }
    }

    pub fn space_if_not_bol(&mut self) {
        if !self.is_bol() {
            self.s.space();
        }
    }

    pub fn break_offset_if_not_bol(&mut self, n: usize, off: isize) {
        if !self.is_bol() {
            self.s.break_offset(n, off)
        } else {
            if off != 0 && self.s.last_token().is_hardbreak_tok() {
                // We do something pretty sketchy here: tuck the nonzero
                // offset-adjustment we were going to deposit along with the
                // break into the previous hardbreak.
                self.s.replace_last_token(pp::Printer::hardbreak_tok_offset(off));
            }
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

    pub fn commasep_cmnt<T, F, G>(&mut self,
                                  b: Breaks,
                                  elts: &[T],
                                  mut op: F,
                                  mut get_span: G)
        where F: FnMut(&mut State<'_>, &T),
              G: FnMut(&T) -> syntax_pos::Span
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

    pub fn commasep_exprs(&mut self, b: Breaks, exprs: &[hir::Expr]) {
        self.commasep_cmnt(b, exprs, |s, e| s.print_expr(&e), |e| e.span)
    }

    pub fn print_mod(&mut self, _mod: &hir::Mod, attrs: &[ast::Attribute]) {
        self.print_inner_attributes(attrs);
        for &item_id in &_mod.item_ids {
            self.ann.nested(self, Nested::Item(item_id));
        }
    }

    pub fn print_foreign_mod(&mut self,
                             nmod: &hir::ForeignMod,
                             attrs: &[ast::Attribute])
                             {
        self.print_inner_attributes(attrs);
        for item in &nmod.items {
            self.print_foreign_item(item);
        }
    }

    pub fn print_opt_lifetime(&mut self, lifetime: &hir::Lifetime) {
        if !lifetime.is_elided() {
            self.print_lifetime(lifetime);
            self.nbsp();
        }
    }

    pub fn print_type(&mut self, ty: &hir::Ty) {
        self.maybe_print_comment(ty.span.lo());
        self.ibox(0);
        match ty.node {
            hir::TyKind::Slice(ref ty) => {
                self.s.word("[");
                self.print_type(&ty);
                self.s.word("]");
            }
            hir::TyKind::Ptr(ref mt) => {
                self.s.word("*");
                match mt.mutbl {
                    hir::MutMutable => self.word_nbsp("mut"),
                    hir::MutImmutable => self.word_nbsp("const"),
                }
                self.print_type(&mt.ty);
            }
            hir::TyKind::Rptr(ref lifetime, ref mt) => {
                self.s.word("&");
                self.print_opt_lifetime(lifetime);
                self.print_mt(mt);
            }
            hir::TyKind::Never => {
                self.s.word("!");
            },
            hir::TyKind::Tup(ref elts) => {
                self.popen();
                self.commasep(Inconsistent, &elts[..], |s, ty| s.print_type(&ty));
                if elts.len() == 1 {
                    self.s.word(",");
                }
                self.pclose();
            }
            hir::TyKind::BareFn(ref f) => {
                self.print_ty_fn(f.abi, f.unsafety, &f.decl, None, &f.generic_params,
                                 &f.arg_names[..]);
            }
            hir::TyKind::Def(..) => {},
            hir::TyKind::Path(ref qpath) => {
                self.print_qpath(qpath, false)
            }
            hir::TyKind::TraitObject(ref bounds, ref lifetime) => {
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
            hir::TyKind::Infer => {
                self.s.word("_");
            }
            hir::TyKind::Err => {
                self.popen();
                self.s.word("/*ERROR*/");
                self.pclose();
            }
            hir::TyKind::CVarArgs(_) => {
                self.s.word("...");
            }
        }
        self.end()
    }

    pub fn print_foreign_item(&mut self, item: &hir::ForeignItem) {
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(item.span.lo());
        self.print_outer_attributes(&item.attrs);
        match item.node {
            hir::ForeignItemKind::Fn(ref decl, ref arg_names, ref generics) => {
                self.head("");
                self.print_fn(decl,
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
                              None);
                self.end(); // end head-ibox
                self.s.word(";");
                self.end() // end the outer fn box
            }
            hir::ForeignItemKind::Static(ref t, m) => {
                self.head(visibility_qualified(&item.vis, "static"));
                if m == hir::MutMutable {
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

    fn print_associated_const(&mut self,
                              ident: ast::Ident,
                              ty: &hir::Ty,
                              default: Option<hir::BodyId>,
                              vis: &hir::Visibility)
                              {
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

    fn print_associated_type(&mut self,
                             ident: ast::Ident,
                             bounds: Option<&hir::GenericBounds>,
                             ty: Option<&hir::Ty>)
                             {
        self.word_space("type");
        self.print_ident(ident);
        if let Some(bounds) = bounds {
            self.print_bounds(":", bounds);
        }
        if let Some(ty) = ty {
            self.s.space();
            self.word_space("=");
            self.print_type(ty);
        }
        self.s.word(";")
    }

    /// Pretty-print an item
    pub fn print_item(&mut self, item: &hir::Item) {
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(item.span.lo());
        self.print_outer_attributes(&item.attrs);
        self.ann.pre(self, AnnNode::Item(item));
        match item.node {
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
                    hir::UseKind::ListStem => self.s.word("::{};")
                }
                self.end(); // end inner head-block
                self.end(); // end outer head-block
            }
            hir::ItemKind::Static(ref ty, m, expr) => {
                self.head(visibility_qualified(&item.vis, "static"));
                if m == hir::MutMutable {
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
            hir::ItemKind::Fn(ref decl, header, ref param_names, body) => {
                self.head("");
                self.print_fn(decl,
                              header,
                              Some(item.ident.name),
                              param_names,
                              &item.vis,
                              &[],
                              Some(body));
                self.s.word(" ");
                self.end(); // need to close a box
                self.end(); // need to close a box
                self.ann.nested(self, Nested::Body(body));
            }
            hir::ItemKind::Mod(ref _mod) => {
                self.head(visibility_qualified(&item.vis, "mod"));
                self.print_ident(item.ident);
                self.nbsp();
                self.bopen();
                self.print_mod(_mod, &item.attrs);
                self.bclose(item.span);
            }
            hir::ItemKind::ForeignMod(ref nmod) => {
                self.head("extern");
                self.word_nbsp(nmod.abi.to_string());
                self.bopen();
                self.print_foreign_mod(nmod, &item.attrs);
                self.bclose(item.span);
            }
            hir::ItemKind::GlobalAsm(ref ga) => {
                self.head(visibility_qualified(&item.vis, "global asm"));
                self.s.word(ga.asm.as_str().to_string());
                self.end()
            }
            hir::ItemKind::Ty(ref ty, ref generics) => {
                self.head(visibility_qualified(&item.vis, "type"));
                self.print_ident(item.ident);
                self.print_generic_params(&generics.params);
                self.end(); // end the inner ibox

                self.print_where_clause(&generics.where_clause);
                self.s.space();
                self.word_space("=");
                self.print_type(&ty);
                self.s.word(";");
                self.end(); // end the outer ibox
            }
            hir::ItemKind::Existential(ref exist) => {
                self.head(visibility_qualified(&item.vis, "existential type"));
                self.print_ident(item.ident);
                self.print_generic_params(&exist.generics.params);
                self.end(); // end the inner ibox

                self.print_where_clause(&exist.generics.where_clause);
                self.s.space();
                let mut real_bounds = Vec::with_capacity(exist.bounds.len());
                for b in exist.bounds.iter() {
                    if let GenericBound::Trait(ref ptr, hir::TraitBoundModifier::Maybe) = *b {
                        self.s.space();
                        self.word_space("for ?");
                        self.print_trait_ref(&ptr.trait_ref);
                    } else {
                        real_bounds.push(b);
                    }
                }
                self.print_bounds(":", real_bounds);
                self.s.word(";");
                self.end(); // end the outer ibox
            }
            hir::ItemKind::Enum(ref enum_definition, ref params) => {
                self.print_enum_def(enum_definition, params, item.ident.name, item.span,
                                    &item.vis);
            }
            hir::ItemKind::Struct(ref struct_def, ref generics) => {
                self.head(visibility_qualified(&item.vis, "struct"));
                self.print_struct(struct_def, generics, item.ident.name, item.span, true);
            }
            hir::ItemKind::Union(ref struct_def, ref generics) => {
                self.head(visibility_qualified(&item.vis, "union"));
                self.print_struct(struct_def, generics, item.ident.name, item.span, true);
            }
            hir::ItemKind::Impl(unsafety,
                          polarity,
                          defaultness,
                          ref generics,
                          ref opt_trait,
                          ref ty,
                          ref impl_items) => {
                self.head("");
                self.print_visibility(&item.vis);
                self.print_defaultness(defaultness);
                self.print_unsafety(unsafety);
                self.word_nbsp("impl");

                if !generics.params.is_empty() {
                    self.print_generic_params(&generics.params);
                    self.s.space();
                }

                if let hir::ImplPolarity::Negative = polarity {
                    self.s.word("!");
                }

                if let Some(ref t) = opt_trait {
                    self.print_trait_ref(t);
                    self.s.space();
                    self.word_space("for");
                }

                self.print_type(&ty);
                self.print_where_clause(&generics.where_clause);

                self.s.space();
                self.bopen();
                self.print_inner_attributes(&item.attrs);
                for impl_item in impl_items {
                    self.ann.nested(self, Nested::ImplItem(impl_item.id));
                }
                self.bclose(item.span);
            }
            hir::ItemKind::Trait(is_auto, unsafety, ref generics, ref bounds, ref trait_items) => {
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

    pub fn print_trait_ref(&mut self, t: &hir::TraitRef) {
        self.print_path(&t.path, false)
    }

    fn print_formal_generic_params(
        &mut self,
        generic_params: &[hir::GenericParam]
    ) {
        if !generic_params.is_empty() {
            self.s.word("for");
            self.print_generic_params(generic_params);
            self.nbsp();
        }
    }

    fn print_poly_trait_ref(&mut self, t: &hir::PolyTraitRef) {
        self.print_formal_generic_params(&t.bound_generic_params);
        self.print_trait_ref(&t.trait_ref)
    }

    pub fn print_enum_def(&mut self,
                          enum_definition: &hir::EnumDef,
                          generics: &hir::Generics,
                          name: ast::Name,
                          span: syntax_pos::Span,
                          visibility: &hir::Visibility)
                          {
        self.head(visibility_qualified(visibility, "enum"));
        self.print_name(name);
        self.print_generic_params(&generics.params);
        self.print_where_clause(&generics.where_clause);
        self.s.space();
        self.print_variants(&enum_definition.variants, span)
    }

    pub fn print_variants(&mut self,
                          variants: &[hir::Variant],
                          span: syntax_pos::Span)
                          {
        self.bopen();
        for v in variants {
            self.space_if_not_bol();
            self.maybe_print_comment(v.span.lo());
            self.print_outer_attributes(&v.node.attrs);
            self.ibox(indent_unit);
            self.print_variant(v);
            self.s.word(",");
            self.end();
            self.maybe_print_trailing_comment(v.span, None);
        }
        self.bclose(span)
    }

    pub fn print_visibility(&mut self, vis: &hir::Visibility) {
        match vis.node {
            hir::VisibilityKind::Public => self.word_nbsp("pub"),
            hir::VisibilityKind::Crate(ast::CrateSugar::JustCrate) => self.word_nbsp("crate"),
            hir::VisibilityKind::Crate(ast::CrateSugar::PubCrate) => self.word_nbsp("pub(crate)"),
            hir::VisibilityKind::Restricted { ref path, .. } => {
                self.s.word("pub(");
                if path.segments.len() == 1 &&
                   path.segments[0].ident.name == kw::Super {
                    // Special case: `super` can print like `pub(super)`.
                    self.s.word("super");
                } else {
                    // Everything else requires `in` at present.
                    self.word_nbsp("in");
                    self.print_path(path, false);
                }
                self.word_nbsp(")");
            }
            hir::VisibilityKind::Inherited => ()
        }
    }

    pub fn print_defaultness(&mut self, defaultness: hir::Defaultness) {
        match defaultness {
            hir::Defaultness::Default { .. } => self.word_nbsp("default"),
            hir::Defaultness::Final => (),
        }
    }

    pub fn print_struct(&mut self,
                        struct_def: &hir::VariantData,
                        generics: &hir::Generics,
                        name: ast::Name,
                        span: syntax_pos::Span,
                        print_finalizer: bool)
                        {
        self.print_name(name);
        self.print_generic_params(&generics.params);
        match struct_def {
            hir::VariantData::Tuple(..) | hir::VariantData::Unit(..) => {
                if let hir::VariantData::Tuple(..) = struct_def {
                    self.popen();
                    self.commasep(Inconsistent, struct_def.fields(), |s, field| {
                        s.maybe_print_comment(field.span.lo());
                        s.print_outer_attributes(&field.attrs);
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
                    self.print_outer_attributes(&field.attrs);
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

    pub fn print_variant(&mut self, v: &hir::Variant) {
        self.head("");
        let generics = hir::Generics::empty();
        self.print_struct(&v.node.data, &generics, v.node.ident.name, v.span, false);
        if let Some(ref d) = v.node.disr_expr {
            self.s.space();
            self.word_space("=");
            self.print_anon_const(d);
        }
    }
    pub fn print_method_sig(&mut self,
                            ident: ast::Ident,
                            m: &hir::MethodSig,
                            generics: &hir::Generics,
                            vis: &hir::Visibility,
                            arg_names: &[ast::Ident],
                            body_id: Option<hir::BodyId>)
                            {
        self.print_fn(&m.decl,
                      m.header,
                      Some(ident.name),
                      generics,
                      vis,
                      arg_names,
                      body_id)
    }

    pub fn print_trait_item(&mut self, ti: &hir::TraitItem) {
        self.ann.pre(self, AnnNode::SubItem(ti.hir_id));
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(ti.span.lo());
        self.print_outer_attributes(&ti.attrs);
        match ti.node {
            hir::TraitItemKind::Const(ref ty, default) => {
                let vis = Spanned { span: syntax_pos::DUMMY_SP,
                                    node: hir::VisibilityKind::Inherited };
                self.print_associated_const(ti.ident, &ty, default, &vis);
            }
            hir::TraitItemKind::Method(ref sig, hir::TraitMethod::Required(ref arg_names)) => {
                let vis = Spanned { span: syntax_pos::DUMMY_SP,
                                    node: hir::VisibilityKind::Inherited };
                self.print_method_sig(ti.ident, sig, &ti.generics, &vis, arg_names, None);
                self.s.word(";");
            }
            hir::TraitItemKind::Method(ref sig, hir::TraitMethod::Provided(body)) => {
                let vis = Spanned { span: syntax_pos::DUMMY_SP,
                                    node: hir::VisibilityKind::Inherited };
                self.head("");
                self.print_method_sig(ti.ident, sig, &ti.generics, &vis, &[], Some(body));
                self.nbsp();
                self.end(); // need to close a box
                self.end(); // need to close a box
                self.ann.nested(self, Nested::Body(body));
            }
            hir::TraitItemKind::Type(ref bounds, ref default) => {
                self.print_associated_type(ti.ident,
                                           Some(bounds),
                                           default.as_ref().map(|ty| &**ty));
            }
        }
        self.ann.post(self, AnnNode::SubItem(ti.hir_id))
    }

    pub fn print_impl_item(&mut self, ii: &hir::ImplItem) {
        self.ann.pre(self, AnnNode::SubItem(ii.hir_id));
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(ii.span.lo());
        self.print_outer_attributes(&ii.attrs);
        self.print_defaultness(ii.defaultness);

        match ii.node {
            hir::ImplItemKind::Const(ref ty, expr) => {
                self.print_associated_const(ii.ident, &ty, Some(expr), &ii.vis);
            }
            hir::ImplItemKind::Method(ref sig, body) => {
                self.head("");
                self.print_method_sig(ii.ident, sig, &ii.generics, &ii.vis, &[], Some(body));
                self.nbsp();
                self.end(); // need to close a box
                self.end(); // need to close a box
                self.ann.nested(self, Nested::Body(body));
            }
            hir::ImplItemKind::Type(ref ty) => {
                self.print_associated_type(ii.ident, None, Some(ty));
            }
            hir::ImplItemKind::Existential(ref bounds) => {
                self.word_space("existential");
                self.print_associated_type(ii.ident, Some(bounds), None);
            }
        }
        self.ann.post(self, AnnNode::SubItem(ii.hir_id))
    }

    pub fn print_local(
        &mut self,
        init: Option<&hir::Expr>,
        decl: impl Fn(&mut Self)
    ) {
        self.space_if_not_bol();
        self.ibox(indent_unit);
        self.word_nbsp("let");

        self.ibox(indent_unit);
        decl(self);
        self.end();

        if let Some(ref init) = init {
            self.nbsp();
            self.word_space("=");
            self.print_expr(&init);
        }
        self.end()
    }

    pub fn print_stmt(&mut self, st: &hir::Stmt) {
        self.maybe_print_comment(st.span.lo());
        match st.node {
            hir::StmtKind::Local(ref loc) => {
                self.print_local(loc.init.deref(), |this| this.print_local_decl(&loc));
            }
            hir::StmtKind::Item(item) => {
                self.ann.nested(self, Nested::Item(item))
            }
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
        if stmt_ends_with_semi(&st.node) {
            self.s.word(";");
        }
        self.maybe_print_trailing_comment(st.span, None)
    }

    pub fn print_block(&mut self, blk: &hir::Block) {
        self.print_block_with_attrs(blk, &[])
    }

    pub fn print_block_unclosed(&mut self, blk: &hir::Block) {
        self.print_block_unclosed_indent(blk, indent_unit)
    }

    pub fn print_block_unclosed_indent(&mut self,
                                       blk: &hir::Block,
                                       indented: usize)
                                       {
        self.print_block_maybe_unclosed(blk, indented, &[], false)
    }

    pub fn print_block_with_attrs(&mut self,
                                  blk: &hir::Block,
                                  attrs: &[ast::Attribute])
                                  {
        self.print_block_maybe_unclosed(blk, indent_unit, attrs, true)
    }

    pub fn print_block_maybe_unclosed(&mut self,
                                      blk: &hir::Block,
                                      indented: usize,
                                      attrs: &[ast::Attribute],
                                      close_box: bool)
                                      {
        match blk.rules {
            hir::UnsafeBlock(..) => self.word_space("unsafe"),
            hir::PushUnsafeBlock(..) => self.word_space("push_unsafe"),
            hir::PopUnsafeBlock(..) => self.word_space("pop_unsafe"),
            hir::DefaultBlock => (),
        }
        self.maybe_print_comment(blk.span.lo());
        self.ann.pre(self, AnnNode::Block(blk));
        self.bopen();

        self.print_inner_attributes(attrs);

        for st in &blk.stmts {
            self.print_stmt(st);
        }
        if let Some(ref expr) = blk.expr {
            self.space_if_not_bol();
            self.print_expr(&expr);
            self.maybe_print_trailing_comment(expr.span, Some(blk.span.hi()));
        }
        self.bclose_maybe_open(blk.span, indented, close_box);
        self.ann.post(self, AnnNode::Block(blk))
    }

    pub fn print_anon_const(&mut self, constant: &hir::AnonConst) {
        self.ann.nested(self, Nested::Body(constant.body))
    }

    fn print_call_post(&mut self, args: &[hir::Expr]) {
        self.popen();
        self.commasep_exprs(Inconsistent, args);
        self.pclose()
    }

    pub fn print_expr_maybe_paren(&mut self, expr: &hir::Expr, prec: i8) {
        let needs_par = expr.precedence().order() < prec;
        if needs_par {
            self.popen();
        }
        self.print_expr(expr);
        if needs_par {
            self.pclose();
        }
    }

    /// Print an expr using syntax that's acceptable in a condition position, such as the `cond` in
    /// `if cond { ... }`.
    pub fn print_expr_as_cond(&mut self, expr: &hir::Expr) {
        let needs_par = match expr.node {
            // These cases need parens due to the parse error observed in #26461: `if return {}`
            // parses as the erroneous construct `if (return {})`, not `if (return) {}`.
            hir::ExprKind::Closure(..) |
            hir::ExprKind::Ret(..) |
            hir::ExprKind::Break(..) => true,

            _ => contains_exterior_struct_lit(expr),
        };

        if needs_par {
            self.popen();
        }
        self.print_expr(expr);
        if needs_par {
            self.pclose();
        }
    }

    fn print_expr_vec(&mut self, exprs: &[hir::Expr]) {
        self.ibox(indent_unit);
        self.s.word("[");
        self.commasep_exprs(Inconsistent, exprs);
        self.s.word("]");
        self.end()
    }

    fn print_expr_repeat(&mut self, element: &hir::Expr, count: &hir::AnonConst) {
        self.ibox(indent_unit);
        self.s.word("[");
        self.print_expr(element);
        self.word_space(";");
        self.print_anon_const(count);
        self.s.word("]");
        self.end()
    }

    fn print_expr_struct(&mut self,
                         qpath: &hir::QPath,
                         fields: &[hir::Field],
                         wth: &Option<P<hir::Expr>>)
                         {
        self.print_qpath(qpath, true);
        self.s.word("{");
        self.commasep_cmnt(Consistent,
                           &fields[..],
                           |s, field| {
                               s.ibox(indent_unit);
                               if !field.is_shorthand {
                                    s.print_ident(field.ident);
                                    s.word_space(":");
                               }
                               s.print_expr(&field.expr);
                               s.end()
                           },
                           |f| f.span);
        match *wth {
            Some(ref expr) => {
                self.ibox(indent_unit);
                if !fields.is_empty() {
                    self.s.word(",");
                    self.s.space();
                }
                self.s.word("..");
                self.print_expr(&expr);
                self.end();
            }
            _ => if !fields.is_empty() {
                self.s.word(",")
            },
        }
        self.s.word("}");
    }

    fn print_expr_tup(&mut self, exprs: &[hir::Expr]) {
        self.popen();
        self.commasep_exprs(Inconsistent, exprs);
        if exprs.len() == 1 {
            self.s.word(",");
        }
        self.pclose()
    }

    fn print_expr_call(&mut self, func: &hir::Expr, args: &[hir::Expr]) {
        let prec =
            match func.node {
                hir::ExprKind::Field(..) => parser::PREC_FORCE_PAREN,
                _ => parser::PREC_POSTFIX,
            };

        self.print_expr_maybe_paren(func, prec);
        self.print_call_post(args)
    }

    fn print_expr_method_call(&mut self,
                              segment: &hir::PathSegment,
                              args: &[hir::Expr])
                              {
        let base_args = &args[1..];
        self.print_expr_maybe_paren(&args[0], parser::PREC_POSTFIX);
        self.s.word(".");
        self.print_ident(segment.ident);

        let generic_args = segment.generic_args();
        if !generic_args.args.is_empty() || !generic_args.bindings.is_empty() {
            self.print_generic_args(generic_args, segment.infer_args, true);
        }

        self.print_call_post(base_args)
    }

    fn print_expr_binary(&mut self,
                         op: hir::BinOp,
                         lhs: &hir::Expr,
                         rhs: &hir::Expr)
                         {
        let assoc_op = bin_op_to_assoc_op(op.node);
        let prec = assoc_op.precedence() as i8;
        let fixity = assoc_op.fixity();

        let (left_prec, right_prec) = match fixity {
            Fixity::Left => (prec, prec + 1),
            Fixity::Right => (prec + 1, prec),
            Fixity::None => (prec + 1, prec + 1),
        };

        let left_prec = match (&lhs.node, op.node) {
            // These cases need parens: `x as i32 < y` has the parser thinking that `i32 < y` is
            // the beginning of a path type. It starts trying to parse `x as (i32 < y ...` instead
            // of `(x as i32) < ...`. We need to convince it _not_ to do that.
            (&hir::ExprKind::Cast { .. }, hir::BinOpKind::Lt) |
            (&hir::ExprKind::Cast { .. }, hir::BinOpKind::Shl) => parser::PREC_FORCE_PAREN,
            _ => left_prec,
        };

        self.print_expr_maybe_paren(lhs, left_prec);
        self.s.space();
        self.word_space(op.node.as_str());
        self.print_expr_maybe_paren(rhs, right_prec)
    }

    fn print_expr_unary(&mut self, op: hir::UnOp, expr: &hir::Expr) {
        self.s.word(op.as_str());
        self.print_expr_maybe_paren(expr, parser::PREC_PREFIX)
    }

    fn print_expr_addr_of(&mut self,
                          mutability: hir::Mutability,
                          expr: &hir::Expr)
                          {
        self.s.word("&");
        self.print_mutability(mutability);
        self.print_expr_maybe_paren(expr, parser::PREC_PREFIX)
    }

    fn print_literal(&mut self, lit: &hir::Lit) {
        self.maybe_print_comment(lit.span.lo());
        self.writer().word(pprust::literal_to_string(lit.node.to_lit_token()))
    }

    pub fn print_expr(&mut self, expr: &hir::Expr) {
        self.maybe_print_comment(expr.span.lo());
        self.print_outer_attributes(&expr.attrs);
        self.ibox(indent_unit);
        self.ann.pre(self, AnnNode::Expr(expr));
        match expr.node {
            hir::ExprKind::Box(ref expr) => {
                self.word_space("box");
                self.print_expr_maybe_paren(expr, parser::PREC_PREFIX);
            }
            hir::ExprKind::Array(ref exprs) => {
                self.print_expr_vec(exprs);
            }
            hir::ExprKind::Repeat(ref element, ref count) => {
                self.print_expr_repeat(&element, count);
            }
            hir::ExprKind::Struct(ref qpath, ref fields, ref wth) => {
                self.print_expr_struct(qpath, &fields[..], wth);
            }
            hir::ExprKind::Tup(ref exprs) => {
                self.print_expr_tup(exprs);
            }
            hir::ExprKind::Call(ref func, ref args) => {
                self.print_expr_call(&func, args);
            }
            hir::ExprKind::MethodCall(ref segment, _, ref args) => {
                self.print_expr_method_call(segment, args);
            }
            hir::ExprKind::Binary(op, ref lhs, ref rhs) => {
                self.print_expr_binary(op, &lhs, &rhs);
            }
            hir::ExprKind::Unary(op, ref expr) => {
                self.print_expr_unary(op, &expr);
            }
            hir::ExprKind::AddrOf(m, ref expr) => {
                self.print_expr_addr_of(m, &expr);
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
                self.cbox(indent_unit);
                self.ibox(0);
                self.bopen();

                // Print `let _t = $init;`:
                let temp = ast::Ident::from_str("_t");
                self.print_local(Some(init), |this| this.print_ident(temp));
                self.s.word(";");

                // Print `_t`:
                self.space_if_not_bol();
                self.print_ident(temp);

                // Print `}`:
                self.bclose_maybe_open(expr.span, indent_unit, true);
            }
            hir::ExprKind::While(ref test, ref blk, opt_label) => {
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }
                self.head("while");
                self.print_expr_as_cond(&test);
                self.s.space();
                self.print_block(&blk);
            }
            hir::ExprKind::Loop(ref blk, opt_label, _) => {
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }
                self.head("loop");
                self.s.space();
                self.print_block(&blk);
            }
            hir::ExprKind::Match(ref expr, ref arms, _) => {
                self.cbox(indent_unit);
                self.ibox(4);
                self.word_nbsp("match");
                self.print_expr_as_cond(&expr);
                self.s.space();
                self.bopen();
                for arm in arms {
                    self.print_arm(arm);
                }
                self.bclose_(expr.span, indent_unit);
            }
            hir::ExprKind::Closure(capture_clause, ref decl, body, _fn_decl_span, _gen) => {
                self.print_capture_clause(capture_clause);

                self.print_closure_args(&decl, body);
                self.s.space();

                // this is a bare expression
                self.ann.nested(self, Nested::Body(body));
                self.end(); // need to close a box

                // a box will be closed by print_expr, but we didn't want an overall
                // wrapper so we closed the corresponding opening. so create an
                // empty box to satisfy the close.
                self.ibox(0);
            }
            hir::ExprKind::Block(ref blk, opt_label) => {
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }
                // containing cbox, will be closed by print-block at }
                self.cbox(indent_unit);
                // head-box, will be closed by print-block after {
                self.ibox(0);
                self.print_block(&blk);
            }
            hir::ExprKind::Assign(ref lhs, ref rhs) => {
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
            hir::ExprKind::Path(ref qpath) => {
                self.print_qpath(qpath, true)
            }
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
            hir::ExprKind::InlineAsm(ref a, ref outputs, ref inputs) => {
                self.s.word("asm!");
                self.popen();
                self.print_string(&a.asm.as_str(), a.asm_str_style);
                self.word_space(":");

                let mut out_idx = 0;
                self.commasep(Inconsistent, &a.outputs, |s, out| {
                    let constraint = out.constraint.as_str();
                    let mut ch = constraint.chars();
                    match ch.next() {
                        Some('=') if out.is_rw => {
                            s.print_string(&format!("+{}", ch.as_str()),
                                           ast::StrStyle::Cooked)
                        }
                        _ => s.print_string(&constraint, ast::StrStyle::Cooked),
                    }
                    s.popen();
                    s.print_expr(&outputs[out_idx]);
                    s.pclose();
                    out_idx += 1;
                });
                self.s.space();
                self.word_space(":");

                let mut in_idx = 0;
                self.commasep(Inconsistent, &a.inputs, |s, co| {
                    s.print_string(&co.as_str(), ast::StrStyle::Cooked);
                    s.popen();
                    s.print_expr(&inputs[in_idx]);
                    s.pclose();
                    in_idx += 1;
                });
                self.s.space();
                self.word_space(":");

                self.commasep(Inconsistent, &a.clobbers, |s, co| {
                    s.print_string(&co.as_str(), ast::StrStyle::Cooked);
                });

                let mut options = vec![];
                if a.volatile {
                    options.push("volatile");
                }
                if a.alignstack {
                    options.push("alignstack");
                }
                if a.dialect == ast::AsmDialect::Intel {
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

    pub fn print_local_decl(&mut self, loc: &hir::Local) {
        self.print_pat(&loc.pat);
        if let Some(ref ty) = loc.ty {
            self.word_space(":");
            self.print_type(&ty);
        }
    }

    pub fn print_usize(&mut self, i: usize) {
        self.s.word(i.to_string())
    }

    pub fn print_ident(&mut self, ident: ast::Ident) {
        if ident.is_raw_guess() {
            self.s.word(format!("r#{}", ident.name));
        } else {
            self.s.word(ident.as_str().to_string());
        }
        self.ann.post(self, AnnNode::Name(&ident.name))
    }

    pub fn print_name(&mut self, name: ast::Name) {
        self.print_ident(ast::Ident::with_empty_ctxt(name))
    }

    pub fn print_for_decl(&mut self, loc: &hir::Local, coll: &hir::Expr) {
        self.print_local_decl(loc);
        self.s.space();
        self.word_space("in");
        self.print_expr(coll)
    }

    pub fn print_path(&mut self,
                      path: &hir::Path,
                      colons_before_params: bool)
                      {
        self.maybe_print_comment(path.span.lo());

        for (i, segment) in path.segments.iter().enumerate() {
            if i > 0 {
                self.s.word("::")
            }
            if segment.ident.name != kw::PathRoot {
                self.print_ident(segment.ident);
                self.print_generic_args(segment.generic_args(), segment.infer_args,
                                        colons_before_params);
            }
        }
    }

    pub fn print_path_segment(&mut self, segment: &hir::PathSegment) {
        if segment.ident.name != kw::PathRoot {
            self.print_ident(segment.ident);
            self.print_generic_args(segment.generic_args(), segment.infer_args, false);
        }
    }

    pub fn print_qpath(&mut self,
                       qpath: &hir::QPath,
                       colons_before_params: bool)
                       {
        match *qpath {
            hir::QPath::Resolved(None, ref path) => {
                self.print_path(path, colons_before_params)
            }
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
                        self.print_generic_args(segment.generic_args(),
                                                segment.infer_args,
                                                colons_before_params);
                    }
                }

                self.s.word(">");
                self.s.word("::");
                let item_segment = path.segments.last().unwrap();
                self.print_ident(item_segment.ident);
                self.print_generic_args(item_segment.generic_args(),
                                        item_segment.infer_args,
                                        colons_before_params)
            }
            hir::QPath::TypeRelative(ref qself, ref item_segment) => {
                self.s.word("<");
                self.print_type(qself);
                self.s.word(">");
                self.s.word("::");
                self.print_ident(item_segment.ident);
                self.print_generic_args(item_segment.generic_args(),
                                        item_segment.infer_args,
                                        colons_before_params)
            }
        }
    }

    fn print_generic_args(&mut self,
                             generic_args: &hir::GenericArgs,
                             infer_args: bool,
                             colons_before_params: bool)
                             {
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
                self.commasep(Inconsistent, &generic_args.args, |s, generic_arg| {
                    match generic_arg {
                        GenericArg::Lifetime(lt) if !elide_lifetimes => s.print_lifetime(lt),
                        GenericArg::Lifetime(_) => {},
                        GenericArg::Type(ty) => s.print_type(ty),
                        GenericArg::Const(ct) => s.print_anon_const(&ct.value),
                    }
                });
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
                self.s.space();
                match generic_args.bindings[0].kind {
                    hir::TypeBindingKind::Equality { ref ty } => {
                        self.word_space("=");
                        self.print_type(ty);
                    }
                    hir::TypeBindingKind::Constraint { ref bounds } => {
                        self.print_bounds(":", bounds);
                    }
                }
            }

            if !empty.get() {
                self.s.word(">")
            }
        }
    }

    pub fn print_pat(&mut self, pat: &hir::Pat) {
        self.maybe_print_comment(pat.span.lo());
        self.ann.pre(self, AnnNode::Pat(pat));
        // Pat isn't normalized, but the beauty of it
        // is that it doesn't matter
        match pat.node {
            PatKind::Wild => self.s.word("_"),
            PatKind::Binding(binding_mode, _, ident, ref sub) => {
                match binding_mode {
                    hir::BindingAnnotation::Ref => {
                        self.word_nbsp("ref");
                        self.print_mutability(hir::MutImmutable);
                    }
                    hir::BindingAnnotation::RefMut => {
                        self.word_nbsp("ref");
                        self.print_mutability(hir::MutMutable);
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
                self.commasep_cmnt(Consistent,
                                   &fields[..],
                                   |s, f| {
                                       s.cbox(indent_unit);
                                       if !f.node.is_shorthand {
                                           s.print_ident(f.node.ident);
                                           s.word_nbsp(":");
                                       }
                                       s.print_pat(&f.node.pat);
                                       s.end()
                                   },
                                   |f| f.node.pat.span);
                if etc {
                    if !fields.is_empty() {
                        self.word_space(",");
                    }
                    self.s.word("..");
                }
                self.s.space();
                self.s.word("}");
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
                let is_range_inner = match inner.node {
                    PatKind::Range(..) => true,
                    _ => false,
                };
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
                let is_range_inner = match inner.node {
                    PatKind::Range(..) => true,
                    _ => false,
                };
                self.s.word("&");
                if mutbl == hir::MutMutable {
                    self.s.word("mut ");
                }
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
                self.print_expr(&begin);
                self.s.space();
                match *end_kind {
                    RangeEnd::Included => self.s.word("..."),
                    RangeEnd::Excluded => self.s.word(".."),
                }
                self.print_expr(&end);
            }
            PatKind::Slice(ref before, ref slice, ref after) => {
                self.s.word("[");
                self.commasep(Inconsistent, &before[..], |s, p| s.print_pat(&p));
                if let Some(ref p) = *slice {
                    if !before.is_empty() {
                        self.word_space(",");
                    }
                    if let PatKind::Wild = p.node {
                        // Print nothing
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

    pub fn print_arm(&mut self, arm: &hir::Arm) {
        // I have no idea why this check is necessary, but here it
        // is :(
        if arm.attrs.is_empty() {
            self.s.space();
        }
        self.cbox(indent_unit);
        self.ibox(0);
        self.print_outer_attributes(&arm.attrs);
        let mut first = true;
        for p in &arm.pats {
            if first {
                first = false;
            } else {
                self.s.space();
                self.word_space("|");
            }
            self.print_pat(&p);
        }
        self.s.space();
        if let Some(ref g) = arm.guard {
            match g {
                hir::Guard::If(e) => {
                    self.word_space("if");
                    self.print_expr(&e);
                    self.s.space();
                }
            }
        }
        self.word_space("=>");

        match arm.body.node {
            hir::ExprKind::Block(ref blk, opt_label) => {
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }
                // the block will close the pattern's ibox
                self.print_block_unclosed_indent(&blk, indent_unit);

                // If it is a user-provided unsafe block, print a comma after it
                if let hir::UnsafeBlock(hir::UserProvided) = blk.rules {
                    self.s.word(",");
                }
            }
            _ => {
                self.end(); // close the ibox for the pattern
                self.print_expr(&arm.body);
                self.s.word(",");
            }
        }
        self.end() // close enclosing cbox
    }

    pub fn print_fn(&mut self,
                    decl: &hir::FnDecl,
                    header: hir::FnHeader,
                    name: Option<ast::Name>,
                    generics: &hir::Generics,
                    vis: &hir::Visibility,
                    arg_names: &[ast::Ident],
                    body_id: Option<hir::BodyId>)
                    {
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
            s.ibox(indent_unit);
            if let Some(arg_name) = arg_names.get(i) {
                s.s.word(arg_name.as_str().to_string());
                s.s.word(":");
                s.s.space();
            } else if let Some(body_id) = body_id {
                s.ann.nested(s, Nested::BodyArgPat(body_id, i));
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

    fn print_closure_args(&mut self, decl: &hir::FnDecl, body_id: hir::BodyId) {
        self.s.word("|");
        let mut i = 0;
        self.commasep(Inconsistent, &decl.inputs, |s, ty| {
            s.ibox(indent_unit);

            s.ann.nested(s, Nested::BodyArgPat(body_id, i));
            i += 1;

            if let hir::TyKind::Infer = ty.node {
                // Print nothing
            } else {
                s.s.word(":");
                s.s.space();
                s.print_type(ty);
            }
            s.end();
        });
        self.s.word("|");

        if let hir::DefaultReturn(..) = decl.output {
            return;
        }

        self.space_if_not_bol();
        self.word_space("->");
        match decl.output {
            hir::Return(ref ty) => {
                self.print_type(&ty);
                self.maybe_print_comment(ty.span.lo())
            }
            hir::DefaultReturn(..) => unreachable!(),
        }
    }

    pub fn print_capture_clause(&mut self, capture_clause: hir::CaptureClause) {
        match capture_clause {
            hir::CaptureByValue => self.word_space("move"),
            hir::CaptureByRef => {},
        }
    }

    pub fn print_bounds<'b>(
        &mut self,
        prefix: &'static str,
        bounds: impl IntoIterator<Item = &'b hir::GenericBound>,
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
                GenericBound::Outlives(lt) => {
                    self.print_lifetime(lt);
                }
            }
        }
    }

    pub fn print_generic_params(&mut self, generic_params: &[GenericParam]) {
        if !generic_params.is_empty() {
            self.s.word("<");

            self.commasep(Inconsistent, generic_params, |s, param| {
                s.print_generic_param(param)
            });

            self.s.word(">");
        }
    }

    pub fn print_generic_param(&mut self, param: &GenericParam) {
        if let GenericParamKind::Const { .. } = param.kind {
            self.word_space("const");
        }

        self.print_ident(param.name.ident());

        match param.kind {
            GenericParamKind::Lifetime { .. } => {
                let mut sep = ":";
                for bound in &param.bounds {
                    match bound {
                        GenericBound::Outlives(lt) => {
                            self.s.word(sep);
                            self.print_lifetime(lt);
                            sep = "+";
                        }
                        _ => bug!(),
                    }
                }
            }
            GenericParamKind::Type { ref default, .. } => {
                self.print_bounds(":", &param.bounds);
                match default {
                    Some(default) => {
                        self.s.space();
                        self.word_space("=");
                        self.print_type(&default)
                    }
                    _ => {}
                }
            }
            GenericParamKind::Const { ref ty } => {
                self.word_space(":");
                self.print_type(ty)
            }
        }
    }

    pub fn print_lifetime(&mut self, lifetime: &hir::Lifetime) {
        self.print_ident(lifetime.name.ident())
    }

    pub fn print_where_clause(&mut self, where_clause: &hir::WhereClause) {
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
                &hir::WherePredicate::BoundPredicate(hir::WhereBoundPredicate {
                    ref bound_generic_params,
                    ref bounded_ty,
                    ref bounds,
                    ..
                }) => {
                    self.print_formal_generic_params(bound_generic_params);
                    self.print_type(&bounded_ty);
                    self.print_bounds(":", bounds);
                }
                &hir::WherePredicate::RegionPredicate(hir::WhereRegionPredicate{ref lifetime,
                                                                                ref bounds,
                                                                                ..}) => {
                    self.print_lifetime(lifetime);
                    self.s.word(":");

                    for (i, bound) in bounds.iter().enumerate() {
                        match bound {
                            GenericBound::Outlives(lt) => {
                                self.print_lifetime(lt);
                            }
                            _ => bug!(),
                        }

                        if i != 0 {
                            self.s.word(":");
                        }
                    }
                }
                &hir::WherePredicate::EqPredicate(hir::WhereEqPredicate{ref lhs_ty,
                                                                        ref rhs_ty,
                                                                        ..}) => {
                    self.print_type(lhs_ty);
                    self.s.space();
                    self.word_space("=");
                    self.print_type(rhs_ty);
                }
            }
        }
    }

    pub fn print_mutability(&mut self, mutbl: hir::Mutability) {
        match mutbl {
            hir::MutMutable => self.word_nbsp("mut"),
            hir::MutImmutable => {},
        }
    }

    pub fn print_mt(&mut self, mt: &hir::MutTy) {
        self.print_mutability(mt.mutbl);
        self.print_type(&mt.ty)
    }

    pub fn print_fn_output(&mut self, decl: &hir::FnDecl) {
        if let hir::DefaultReturn(..) = decl.output {
            return;
        }

        self.space_if_not_bol();
        self.ibox(indent_unit);
        self.word_space("->");
        match decl.output {
            hir::DefaultReturn(..) => unreachable!(),
            hir::Return(ref ty) => self.print_type(&ty),
        }
        self.end();

        match decl.output {
            hir::Return(ref output) => self.maybe_print_comment(output.span.lo()),
            _ => {},
        }
    }

    pub fn print_ty_fn(&mut self,
                       abi: Abi,
                       unsafety: hir::Unsafety,
                       decl: &hir::FnDecl,
                       name: Option<ast::Name>,
                       generic_params: &[hir::GenericParam],
                       arg_names: &[ast::Ident])
                       {
        self.ibox(indent_unit);
        if !generic_params.is_empty() {
            self.s.word("for");
            self.print_generic_params(generic_params);
        }
        let generics = hir::Generics {
            params: hir::HirVec::new(),
            where_clause: hir::WhereClause {
                predicates: hir::HirVec::new(),
                span: syntax_pos::DUMMY_SP,
            },
            span: syntax_pos::DUMMY_SP,
        };
        self.print_fn(decl,
                      hir::FnHeader {
                          unsafety,
                          abi,
                          constness: hir::Constness::NotConst,
                          asyncness: hir::IsAsync::NotAsync,
                      },
                      name,
                      &generics,
                      &Spanned { span: syntax_pos::DUMMY_SP,
                                 node: hir::VisibilityKind::Inherited },
                      arg_names,
                      None);
        self.end();
    }

    pub fn maybe_print_trailing_comment(&mut self,
                                        span: syntax_pos::Span,
                                        next_pos: Option<BytePos>)
                                        {
        let cm = match self.cm {
            Some(cm) => cm,
            _ => return,
        };
        if let Some(ref cmnt) = self.next_comment() {
            if (*cmnt).style != comments::Trailing {
                return;
            }
            let span_line = cm.lookup_char_pos(span.hi());
            let comment_line = cm.lookup_char_pos((*cmnt).pos);
            let mut next = (*cmnt).pos + BytePos(1);
            if let Some(p) = next_pos {
                next = p;
            }
            if span.hi() < (*cmnt).pos && (*cmnt).pos < next &&
               span_line.line == comment_line.line {
                self.print_comment(cmnt);
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

    pub fn print_opt_abi_and_extern_if_nondefault(&mut self,
                                                  opt_abi: Option<Abi>)
                                                  {
        match opt_abi {
            Some(Abi::Rust) => {},
            Some(abi) => {
                self.word_nbsp("extern");
                self.word_nbsp(abi.to_string())
            }
            None => {},
        }
    }

    pub fn print_extern_opt_abi(&mut self, opt_abi: Option<Abi>) {
        match opt_abi {
            Some(abi) => {
                self.word_nbsp("extern");
                self.word_nbsp(abi.to_string())
            }
            None => {},
        }
    }

    pub fn print_fn_header_info(&mut self,
                                header: hir::FnHeader,
                                vis: &hir::Visibility)
                                {
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
            hir::IsAuto::No => {},
        }
    }
}

// Dup'ed from parse::classify, but adapted for the HIR.
/// Does this expression require a semicolon to be treated
/// as a statement? The negation of this: 'can this expression
/// be used as a statement without a semicolon' -- is used
/// as an early-bail-out in the parser so that, for instance,
///     if true {...} else {...}
///      |x| 5
/// isn't parsed as (if true {...} else {...} | x) | 5
fn expr_requires_semi_to_be_stmt(e: &hir::Expr) -> bool {
    match e.node {
        hir::ExprKind::Match(..) |
        hir::ExprKind::Block(..) |
        hir::ExprKind::While(..) |
        hir::ExprKind::Loop(..) => false,
        _ => true,
    }
}

/// this statement requires a semicolon after it.
/// note that in one case (stmt_semi), we've already
/// seen the semicolon, and thus don't need another.
fn stmt_ends_with_semi(stmt: &hir::StmtKind) -> bool {
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

/// Expressions that syntactically contain an "exterior" struct literal i.e., not surrounded by any
/// parens or other delimiters, e.g., `X { y: 1 }`, `X { y: 1 }.method()`, `foo == X { y: 1 }` and
/// `X { y: 1 } == foo` all do, but `(X { y: 1 }) == foo` does not.
fn contains_exterior_struct_lit(value: &hir::Expr) -> bool {
    match value.node {
        hir::ExprKind::Struct(..) => true,

        hir::ExprKind::Assign(ref lhs, ref rhs) |
        hir::ExprKind::AssignOp(_, ref lhs, ref rhs) |
        hir::ExprKind::Binary(_, ref lhs, ref rhs) => {
            // X { y: 1 } + X { y: 2 }
            contains_exterior_struct_lit(&lhs) || contains_exterior_struct_lit(&rhs)
        }
        hir::ExprKind::Unary(_, ref x) |
        hir::ExprKind::Cast(ref x, _) |
        hir::ExprKind::Type(ref x, _) |
        hir::ExprKind::Field(ref x, _) |
        hir::ExprKind::Index(ref x, _) => {
            // &X { y: 1 }, X { y: 1 }.y
            contains_exterior_struct_lit(&x)
        }

        hir::ExprKind::MethodCall(.., ref exprs) => {
            // X { y: 1 }.bar(...)
            contains_exterior_struct_lit(&exprs[0])
        }

        _ => false,
    }
}
