// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::AnnNode::*;

use syntax::abi::Abi;
use syntax::ast;
use syntax::codemap::{CodeMap, Spanned};
use syntax::parse::ParseSess;
use syntax::parse::lexer::comments;
use syntax::print::pp::{self, break_offset, word, space, hardbreak};
use syntax::print::pp::{Breaks, eof};
use syntax::print::pp::Breaks::{Consistent, Inconsistent};
use syntax::print::pprust::{self as ast_pp, PrintState};
use syntax::ptr::P;
use syntax::symbol::keywords;
use syntax_pos::{self, BytePos};

use hir;
use hir::{PatKind, RegionTyParamBound, TraitTyParamBound, TraitBoundModifier};

use std::io::{self, Write, Read};

pub enum AnnNode<'a> {
    NodeName(&'a ast::Name),
    NodeBlock(&'a hir::Block),
    NodeItem(&'a hir::Item),
    NodeSubItem(ast::NodeId),
    NodeExpr(&'a hir::Expr),
    NodePat(&'a hir::Pat),
}

pub enum Nested {
    Item(hir::ItemId),
    TraitItem(hir::TraitItemId),
    ImplItem(hir::ImplItemId),
    Body(hir::BodyId),
    BodyArgPat(hir::BodyId, usize)
}

pub trait PpAnn {
    fn nested(&self, _state: &mut State, _nested: Nested) -> io::Result<()> {
        Ok(())
    }
    fn pre(&self, _state: &mut State, _node: AnnNode) -> io::Result<()> {
        Ok(())
    }
    fn post(&self, _state: &mut State, _node: AnnNode) -> io::Result<()> {
        Ok(())
    }
}

pub struct NoAnn;
impl PpAnn for NoAnn {}
pub const NO_ANN: &'static PpAnn = &NoAnn;

impl PpAnn for hir::Crate {
    fn nested(&self, state: &mut State, nested: Nested) -> io::Result<()> {
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
    cm: Option<&'a CodeMap>,
    comments: Option<Vec<comments::Comment>>,
    literals: Option<Vec<comments::Literal>>,
    cur_cmnt_and_lit: ast_pp::CurrentCommentAndLiteral,
    boxes: Vec<pp::Breaks>,
    ann: &'a (PpAnn + 'a),
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

    fn cur_cmnt_and_lit(&mut self) -> &mut ast_pp::CurrentCommentAndLiteral {
        &mut self.cur_cmnt_and_lit
    }

    fn literals(&self) -> &Option<Vec<comments::Literal>> {
        &self.literals
    }
}

#[allow(non_upper_case_globals)]
pub const indent_unit: usize = 4;

#[allow(non_upper_case_globals)]
pub const default_columns: usize = 78;


/// Requires you to pass an input filename and reader so that
/// it can scan the input text for comments and literals to
/// copy forward.
pub fn print_crate<'a>(cm: &'a CodeMap,
                       sess: &ParseSess,
                       krate: &hir::Crate,
                       filename: String,
                       input: &mut Read,
                       out: Box<Write + 'a>,
                       ann: &'a PpAnn,
                       is_expanded: bool)
                       -> io::Result<()> {
    let mut s = State::new_from_input(cm, sess, filename, input, out, ann, is_expanded);

    // When printing the AST, we sometimes need to inject `#[no_std]` here.
    // Since you can't compile the HIR, it's not necessary.

    s.print_mod(&krate.module, &krate.attrs)?;
    s.print_remaining_comments()?;
    eof(&mut s.s)
}

impl<'a> State<'a> {
    pub fn new_from_input(cm: &'a CodeMap,
                          sess: &ParseSess,
                          filename: String,
                          input: &mut Read,
                          out: Box<Write + 'a>,
                          ann: &'a PpAnn,
                          is_expanded: bool)
                          -> State<'a> {
        let (cmnts, lits) = comments::gather_comments_and_literals(sess, filename, input);

        State::new(cm,
                   out,
                   ann,
                   Some(cmnts),
                   // If the code is post expansion, don't use the table of
                   // literals, since it doesn't correspond with the literals
                   // in the AST anymore.
                   if is_expanded {
                       None
                   } else {
                       Some(lits)
                   })
    }

    pub fn new(cm: &'a CodeMap,
               out: Box<Write + 'a>,
               ann: &'a PpAnn,
               comments: Option<Vec<comments::Comment>>,
               literals: Option<Vec<comments::Literal>>)
               -> State<'a> {
        State {
            s: pp::mk_printer(out, default_columns),
            cm: Some(cm),
            comments: comments.clone(),
            literals: literals.clone(),
            cur_cmnt_and_lit: ast_pp::CurrentCommentAndLiteral {
                cur_cmnt: 0,
                cur_lit: 0,
            },
            boxes: Vec::new(),
            ann: ann,
        }
    }
}

pub fn to_string<F>(ann: &PpAnn, f: F) -> String
    where F: FnOnce(&mut State) -> io::Result<()>
{
    let mut wr = Vec::new();
    {
        let mut printer = State {
            s: pp::mk_printer(Box::new(&mut wr), default_columns),
            cm: None,
            comments: None,
            literals: None,
            cur_cmnt_and_lit: ast_pp::CurrentCommentAndLiteral {
                cur_cmnt: 0,
                cur_lit: 0,
            },
            boxes: Vec::new(),
            ann: ann,
        };
        f(&mut printer).unwrap();
        eof(&mut printer.s).unwrap();
    }
    String::from_utf8(wr).unwrap()
}

pub fn visibility_qualified(vis: &hir::Visibility, w: &str) -> String {
    to_string(NO_ANN, |s| {
        s.print_visibility(vis)?;
        word(&mut s.s, w)
    })
}

fn needs_parentheses(expr: &hir::Expr) -> bool {
    match expr.node {
        hir::ExprAssign(..) |
        hir::ExprBinary(..) |
        hir::ExprClosure(..) |
        hir::ExprAssignOp(..) |
        hir::ExprCast(..) |
        hir::ExprType(..) => true,
        _ => false,
    }
}

impl<'a> State<'a> {
    pub fn cbox(&mut self, u: usize) -> io::Result<()> {
        self.boxes.push(pp::Breaks::Consistent);
        pp::cbox(&mut self.s, u)
    }

    pub fn nbsp(&mut self) -> io::Result<()> {
        word(&mut self.s, " ")
    }

    pub fn word_nbsp(&mut self, w: &str) -> io::Result<()> {
        word(&mut self.s, w)?;
        self.nbsp()
    }

    pub fn head(&mut self, w: &str) -> io::Result<()> {
        // outer-box is consistent
        self.cbox(indent_unit)?;
        // head-box is inconsistent
        self.ibox(w.len() + 1)?;
        // keyword that starts the head
        if !w.is_empty() {
            self.word_nbsp(w)?;
        }
        Ok(())
    }

    pub fn bopen(&mut self) -> io::Result<()> {
        word(&mut self.s, "{")?;
        self.end() // close the head-box
    }

    pub fn bclose_(&mut self, span: syntax_pos::Span, indented: usize) -> io::Result<()> {
        self.bclose_maybe_open(span, indented, true)
    }
    pub fn bclose_maybe_open(&mut self,
                             span: syntax_pos::Span,
                             indented: usize,
                             close_box: bool)
                             -> io::Result<()> {
        self.maybe_print_comment(span.hi)?;
        self.break_offset_if_not_bol(1, -(indented as isize))?;
        word(&mut self.s, "}")?;
        if close_box {
            self.end()?; // close the outer-box
        }
        Ok(())
    }
    pub fn bclose(&mut self, span: syntax_pos::Span) -> io::Result<()> {
        self.bclose_(span, indent_unit)
    }

    pub fn in_cbox(&self) -> bool {
        match self.boxes.last() {
            Some(&last_box) => last_box == pp::Breaks::Consistent,
            None => false,
        }
    }
    pub fn space_if_not_bol(&mut self) -> io::Result<()> {
        if !self.is_bol() {
            space(&mut self.s)?;
        }
        Ok(())
    }
    pub fn break_offset_if_not_bol(&mut self, n: usize, off: isize) -> io::Result<()> {
        if !self.is_bol() {
            break_offset(&mut self.s, n, off)
        } else {
            if off != 0 && self.s.last_token().is_hardbreak_tok() {
                // We do something pretty sketchy here: tuck the nonzero
                // offset-adjustment we were going to deposit along with the
                // break into the previous hardbreak.
                self.s.replace_last_token(pp::hardbreak_tok_offset(off));
            }
            Ok(())
        }
    }

    // Synthesizes a comment that was not textually present in the original source
    // file.
    pub fn synth_comment(&mut self, text: String) -> io::Result<()> {
        word(&mut self.s, "/*")?;
        space(&mut self.s)?;
        word(&mut self.s, &text[..])?;
        space(&mut self.s)?;
        word(&mut self.s, "*/")
    }


    pub fn commasep_cmnt<T, F, G>(&mut self,
                                  b: Breaks,
                                  elts: &[T],
                                  mut op: F,
                                  mut get_span: G)
                                  -> io::Result<()>
        where F: FnMut(&mut State, &T) -> io::Result<()>,
              G: FnMut(&T) -> syntax_pos::Span
    {
        self.rbox(0, b)?;
        let len = elts.len();
        let mut i = 0;
        for elt in elts {
            self.maybe_print_comment(get_span(elt).hi)?;
            op(self, elt)?;
            i += 1;
            if i < len {
                word(&mut self.s, ",")?;
                self.maybe_print_trailing_comment(get_span(elt), Some(get_span(&elts[i]).hi))?;
                self.space_if_not_bol()?;
            }
        }
        self.end()
    }

    pub fn commasep_exprs(&mut self, b: Breaks, exprs: &[hir::Expr]) -> io::Result<()> {
        self.commasep_cmnt(b, exprs, |s, e| s.print_expr(&e), |e| e.span)
    }

    pub fn print_mod(&mut self, _mod: &hir::Mod, attrs: &[ast::Attribute]) -> io::Result<()> {
        self.print_inner_attributes(attrs)?;
        for &item_id in &_mod.item_ids {
            self.ann.nested(self, Nested::Item(item_id))?;
        }
        Ok(())
    }

    pub fn print_foreign_mod(&mut self,
                             nmod: &hir::ForeignMod,
                             attrs: &[ast::Attribute])
                             -> io::Result<()> {
        self.print_inner_attributes(attrs)?;
        for item in &nmod.items {
            self.print_foreign_item(item)?;
        }
        Ok(())
    }

    pub fn print_opt_lifetime(&mut self, lifetime: &Option<hir::Lifetime>) -> io::Result<()> {
        if let Some(l) = *lifetime {
            self.print_lifetime(&l)?;
            self.nbsp()?;
        }
        Ok(())
    }

    pub fn print_type(&mut self, ty: &hir::Ty) -> io::Result<()> {
        self.maybe_print_comment(ty.span.lo)?;
        self.ibox(0)?;
        match ty.node {
            hir::TySlice(ref ty) => {
                word(&mut self.s, "[")?;
                self.print_type(&ty)?;
                word(&mut self.s, "]")?;
            }
            hir::TyPtr(ref mt) => {
                word(&mut self.s, "*")?;
                match mt.mutbl {
                    hir::MutMutable => self.word_nbsp("mut")?,
                    hir::MutImmutable => self.word_nbsp("const")?,
                }
                self.print_type(&mt.ty)?;
            }
            hir::TyRptr(ref lifetime, ref mt) => {
                word(&mut self.s, "&")?;
                self.print_opt_lifetime(lifetime)?;
                self.print_mt(mt)?;
            }
            hir::TyNever => {
                word(&mut self.s, "!")?;
            },
            hir::TyTup(ref elts) => {
                self.popen()?;
                self.commasep(Inconsistent, &elts[..], |s, ty| s.print_type(&ty))?;
                if elts.len() == 1 {
                    word(&mut self.s, ",")?;
                }
                self.pclose()?;
            }
            hir::TyBareFn(ref f) => {
                let generics = hir::Generics {
                    lifetimes: f.lifetimes.clone(),
                    ty_params: hir::HirVec::new(),
                    where_clause: hir::WhereClause {
                        id: ast::DUMMY_NODE_ID,
                        predicates: hir::HirVec::new(),
                    },
                    span: syntax_pos::DUMMY_SP,
                };
                self.print_ty_fn(f.abi, f.unsafety, &f.decl, None, &generics)?;
            }
            hir::TyPath(ref qpath) => {
                self.print_qpath(qpath, false)?
            }
            hir::TyTraitObject(ref bounds) => {
                self.print_bounds("", &bounds[..])?;
            }
            hir::TyImplTrait(ref bounds) => {
                self.print_bounds("impl ", &bounds[..])?;
            }
            hir::TyArray(ref ty, v) => {
                word(&mut self.s, "[")?;
                self.print_type(&ty)?;
                word(&mut self.s, "; ")?;
                self.ann.nested(self, Nested::Body(v))?;
                word(&mut self.s, "]")?;
            }
            hir::TyTypeof(e) => {
                word(&mut self.s, "typeof(")?;
                self.ann.nested(self, Nested::Body(e))?;
                word(&mut self.s, ")")?;
            }
            hir::TyInfer => {
                word(&mut self.s, "_")?;
            }
        }
        self.end()
    }

    pub fn print_foreign_item(&mut self, item: &hir::ForeignItem) -> io::Result<()> {
        self.hardbreak_if_not_bol()?;
        self.maybe_print_comment(item.span.lo)?;
        self.print_outer_attributes(&item.attrs)?;
        match item.node {
            hir::ForeignItemFn(ref decl, ref arg_names, ref generics) => {
                self.head("")?;
                self.print_fn(decl,
                              hir::Unsafety::Normal,
                              hir::Constness::NotConst,
                              Abi::Rust,
                              Some(item.name),
                              generics,
                              &item.vis,
                              arg_names,
                              None)?;
                self.end()?; // end head-ibox
                word(&mut self.s, ";")?;
                self.end() // end the outer fn box
            }
            hir::ForeignItemStatic(ref t, m) => {
                self.head(&visibility_qualified(&item.vis, "static"))?;
                if m {
                    self.word_space("mut")?;
                }
                self.print_name(item.name)?;
                self.word_space(":")?;
                self.print_type(&t)?;
                word(&mut self.s, ";")?;
                self.end()?; // end the head-ibox
                self.end() // end the outer cbox
            }
        }
    }

    fn print_associated_const(&mut self,
                              name: ast::Name,
                              ty: &hir::Ty,
                              default: Option<hir::BodyId>,
                              vis: &hir::Visibility)
                              -> io::Result<()> {
        word(&mut self.s, &visibility_qualified(vis, ""))?;
        self.word_space("const")?;
        self.print_name(name)?;
        self.word_space(":")?;
        self.print_type(ty)?;
        if let Some(expr) = default {
            space(&mut self.s)?;
            self.word_space("=")?;
            self.ann.nested(self, Nested::Body(expr))?;
        }
        word(&mut self.s, ";")
    }

    fn print_associated_type(&mut self,
                             name: ast::Name,
                             bounds: Option<&hir::TyParamBounds>,
                             ty: Option<&hir::Ty>)
                             -> io::Result<()> {
        self.word_space("type")?;
        self.print_name(name)?;
        if let Some(bounds) = bounds {
            self.print_bounds(":", bounds)?;
        }
        if let Some(ty) = ty {
            space(&mut self.s)?;
            self.word_space("=")?;
            self.print_type(ty)?;
        }
        word(&mut self.s, ";")
    }

    /// Pretty-print an item
    pub fn print_item(&mut self, item: &hir::Item) -> io::Result<()> {
        self.hardbreak_if_not_bol()?;
        self.maybe_print_comment(item.span.lo)?;
        self.print_outer_attributes(&item.attrs)?;
        self.ann.pre(self, NodeItem(item))?;
        match item.node {
            hir::ItemExternCrate(ref optional_path) => {
                self.head(&visibility_qualified(&item.vis, "extern crate"))?;
                if let Some(p) = *optional_path {
                    let val = p.as_str();
                    if val.contains("-") {
                        self.print_string(&val, ast::StrStyle::Cooked)?;
                    } else {
                        self.print_name(p)?;
                    }
                    space(&mut self.s)?;
                    word(&mut self.s, "as")?;
                    space(&mut self.s)?;
                }
                self.print_name(item.name)?;
                word(&mut self.s, ";")?;
                self.end()?; // end inner head-block
                self.end()?; // end outer head-block
            }
            hir::ItemUse(ref path, kind) => {
                self.head(&visibility_qualified(&item.vis, "use"))?;
                self.print_path(path, false)?;

                match kind {
                    hir::UseKind::Single => {
                        if path.segments.last().unwrap().name != item.name {
                            space(&mut self.s)?;
                            self.word_space("as")?;
                            self.print_name(item.name)?;
                        }
                        word(&mut self.s, ";")?;
                    }
                    hir::UseKind::Glob => word(&mut self.s, "::*;")?,
                    hir::UseKind::ListStem => word(&mut self.s, "::{};")?
                }
                self.end()?; // end inner head-block
                self.end()?; // end outer head-block
            }
            hir::ItemStatic(ref ty, m, expr) => {
                self.head(&visibility_qualified(&item.vis, "static"))?;
                if m == hir::MutMutable {
                    self.word_space("mut")?;
                }
                self.print_name(item.name)?;
                self.word_space(":")?;
                self.print_type(&ty)?;
                space(&mut self.s)?;
                self.end()?; // end the head-ibox

                self.word_space("=")?;
                self.ann.nested(self, Nested::Body(expr))?;
                word(&mut self.s, ";")?;
                self.end()?; // end the outer cbox
            }
            hir::ItemConst(ref ty, expr) => {
                self.head(&visibility_qualified(&item.vis, "const"))?;
                self.print_name(item.name)?;
                self.word_space(":")?;
                self.print_type(&ty)?;
                space(&mut self.s)?;
                self.end()?; // end the head-ibox

                self.word_space("=")?;
                self.ann.nested(self, Nested::Body(expr))?;
                word(&mut self.s, ";")?;
                self.end()?; // end the outer cbox
            }
            hir::ItemFn(ref decl, unsafety, constness, abi, ref typarams, body) => {
                self.head("")?;
                self.print_fn(decl,
                              unsafety,
                              constness,
                              abi,
                              Some(item.name),
                              typarams,
                              &item.vis,
                              &[],
                              Some(body))?;
                word(&mut self.s, " ")?;
                self.end()?; // need to close a box
                self.end()?; // need to close a box
                self.ann.nested(self, Nested::Body(body))?;
            }
            hir::ItemMod(ref _mod) => {
                self.head(&visibility_qualified(&item.vis, "mod"))?;
                self.print_name(item.name)?;
                self.nbsp()?;
                self.bopen()?;
                self.print_mod(_mod, &item.attrs)?;
                self.bclose(item.span)?;
            }
            hir::ItemForeignMod(ref nmod) => {
                self.head("extern")?;
                self.word_nbsp(&nmod.abi.to_string())?;
                self.bopen()?;
                self.print_foreign_mod(nmod, &item.attrs)?;
                self.bclose(item.span)?;
            }
            hir::ItemTy(ref ty, ref params) => {
                self.ibox(indent_unit)?;
                self.ibox(0)?;
                self.word_nbsp(&visibility_qualified(&item.vis, "type"))?;
                self.print_name(item.name)?;
                self.print_generics(params)?;
                self.end()?; // end the inner ibox

                self.print_where_clause(&params.where_clause)?;
                space(&mut self.s)?;
                self.word_space("=")?;
                self.print_type(&ty)?;
                word(&mut self.s, ";")?;
                self.end()?; // end the outer ibox
            }
            hir::ItemEnum(ref enum_definition, ref params) => {
                self.print_enum_def(enum_definition, params, item.name, item.span, &item.vis)?;
            }
            hir::ItemStruct(ref struct_def, ref generics) => {
                self.head(&visibility_qualified(&item.vis, "struct"))?;
                self.print_struct(struct_def, generics, item.name, item.span, true)?;
            }
            hir::ItemUnion(ref struct_def, ref generics) => {
                self.head(&visibility_qualified(&item.vis, "union"))?;
                self.print_struct(struct_def, generics, item.name, item.span, true)?;
            }
            hir::ItemDefaultImpl(unsafety, ref trait_ref) => {
                self.head("")?;
                self.print_visibility(&item.vis)?;
                self.print_unsafety(unsafety)?;
                self.word_nbsp("impl")?;
                self.print_trait_ref(trait_ref)?;
                space(&mut self.s)?;
                self.word_space("for")?;
                self.word_space("..")?;
                self.bopen()?;
                self.bclose(item.span)?;
            }
            hir::ItemImpl(unsafety,
                          polarity,
                          ref generics,
                          ref opt_trait,
                          ref ty,
                          ref impl_items) => {
                self.head("")?;
                self.print_visibility(&item.vis)?;
                self.print_unsafety(unsafety)?;
                self.word_nbsp("impl")?;

                if generics.is_parameterized() {
                    self.print_generics(generics)?;
                    space(&mut self.s)?;
                }

                match polarity {
                    hir::ImplPolarity::Negative => {
                        word(&mut self.s, "!")?;
                    }
                    _ => {}
                }

                match opt_trait {
                    &Some(ref t) => {
                        self.print_trait_ref(t)?;
                        space(&mut self.s)?;
                        self.word_space("for")?;
                    }
                    &None => {}
                }

                self.print_type(&ty)?;
                self.print_where_clause(&generics.where_clause)?;

                space(&mut self.s)?;
                self.bopen()?;
                self.print_inner_attributes(&item.attrs)?;
                for impl_item in impl_items {
                    self.ann.nested(self, Nested::ImplItem(impl_item.id))?;
                }
                self.bclose(item.span)?;
            }
            hir::ItemTrait(unsafety, ref generics, ref bounds, ref trait_items) => {
                self.head("")?;
                self.print_visibility(&item.vis)?;
                self.print_unsafety(unsafety)?;
                self.word_nbsp("trait")?;
                self.print_name(item.name)?;
                self.print_generics(generics)?;
                let mut real_bounds = Vec::with_capacity(bounds.len());
                for b in bounds.iter() {
                    if let TraitTyParamBound(ref ptr, hir::TraitBoundModifier::Maybe) = *b {
                        space(&mut self.s)?;
                        self.word_space("for ?")?;
                        self.print_trait_ref(&ptr.trait_ref)?;
                    } else {
                        real_bounds.push(b.clone());
                    }
                }
                self.print_bounds(":", &real_bounds[..])?;
                self.print_where_clause(&generics.where_clause)?;
                word(&mut self.s, " ")?;
                self.bopen()?;
                for trait_item in trait_items {
                    self.ann.nested(self, Nested::TraitItem(trait_item.id))?;
                }
                self.bclose(item.span)?;
            }
        }
        self.ann.post(self, NodeItem(item))
    }

    pub fn print_trait_ref(&mut self, t: &hir::TraitRef) -> io::Result<()> {
        self.print_path(&t.path, false)
    }

    fn print_formal_lifetime_list(&mut self, lifetimes: &[hir::LifetimeDef]) -> io::Result<()> {
        if !lifetimes.is_empty() {
            word(&mut self.s, "for<")?;
            let mut comma = false;
            for lifetime_def in lifetimes {
                if comma {
                    self.word_space(",")?
                }
                self.print_lifetime_def(lifetime_def)?;
                comma = true;
            }
            word(&mut self.s, ">")?;
        }
        Ok(())
    }

    fn print_poly_trait_ref(&mut self, t: &hir::PolyTraitRef) -> io::Result<()> {
        self.print_formal_lifetime_list(&t.bound_lifetimes)?;
        self.print_trait_ref(&t.trait_ref)
    }

    pub fn print_enum_def(&mut self,
                          enum_definition: &hir::EnumDef,
                          generics: &hir::Generics,
                          name: ast::Name,
                          span: syntax_pos::Span,
                          visibility: &hir::Visibility)
                          -> io::Result<()> {
        self.head(&visibility_qualified(visibility, "enum"))?;
        self.print_name(name)?;
        self.print_generics(generics)?;
        self.print_where_clause(&generics.where_clause)?;
        space(&mut self.s)?;
        self.print_variants(&enum_definition.variants, span)
    }

    pub fn print_variants(&mut self,
                          variants: &[hir::Variant],
                          span: syntax_pos::Span)
                          -> io::Result<()> {
        self.bopen()?;
        for v in variants {
            self.space_if_not_bol()?;
            self.maybe_print_comment(v.span.lo)?;
            self.print_outer_attributes(&v.node.attrs)?;
            self.ibox(indent_unit)?;
            self.print_variant(v)?;
            word(&mut self.s, ",")?;
            self.end()?;
            self.maybe_print_trailing_comment(v.span, None)?;
        }
        self.bclose(span)
    }

    pub fn print_visibility(&mut self, vis: &hir::Visibility) -> io::Result<()> {
        match *vis {
            hir::Public => self.word_nbsp("pub"),
            hir::Visibility::Crate => self.word_nbsp("pub(crate)"),
            hir::Visibility::Restricted { ref path, .. } => {
                word(&mut self.s, "pub(")?;
                self.print_path(path, false)?;
                self.word_nbsp(")")
            }
            hir::Inherited => Ok(()),
        }
    }

    pub fn print_struct(&mut self,
                        struct_def: &hir::VariantData,
                        generics: &hir::Generics,
                        name: ast::Name,
                        span: syntax_pos::Span,
                        print_finalizer: bool)
                        -> io::Result<()> {
        self.print_name(name)?;
        self.print_generics(generics)?;
        if !struct_def.is_struct() {
            if struct_def.is_tuple() {
                self.popen()?;
                self.commasep(Inconsistent, struct_def.fields(), |s, field| {
                    s.maybe_print_comment(field.span.lo)?;
                    s.print_outer_attributes(&field.attrs)?;
                    s.print_visibility(&field.vis)?;
                    s.print_type(&field.ty)
                })?;
                self.pclose()?;
            }
            self.print_where_clause(&generics.where_clause)?;
            if print_finalizer {
                word(&mut self.s, ";")?;
            }
            self.end()?;
            self.end() // close the outer-box
        } else {
            self.print_where_clause(&generics.where_clause)?;
            self.nbsp()?;
            self.bopen()?;
            self.hardbreak_if_not_bol()?;

            for field in struct_def.fields() {
                self.hardbreak_if_not_bol()?;
                self.maybe_print_comment(field.span.lo)?;
                self.print_outer_attributes(&field.attrs)?;
                self.print_visibility(&field.vis)?;
                self.print_name(field.name)?;
                self.word_nbsp(":")?;
                self.print_type(&field.ty)?;
                word(&mut self.s, ",")?;
            }

            self.bclose(span)
        }
    }

    pub fn print_variant(&mut self, v: &hir::Variant) -> io::Result<()> {
        self.head("")?;
        let generics = hir::Generics::empty();
        self.print_struct(&v.node.data, &generics, v.node.name, v.span, false)?;
        if let Some(d) = v.node.disr_expr {
            space(&mut self.s)?;
            self.word_space("=")?;
            self.ann.nested(self, Nested::Body(d))?;
        }
        Ok(())
    }
    pub fn print_method_sig(&mut self,
                            name: ast::Name,
                            m: &hir::MethodSig,
                            vis: &hir::Visibility,
                            arg_names: &[Spanned<ast::Name>],
                            body_id: Option<hir::BodyId>)
                            -> io::Result<()> {
        self.print_fn(&m.decl,
                      m.unsafety,
                      m.constness,
                      m.abi,
                      Some(name),
                      &m.generics,
                      vis,
                      arg_names,
                      body_id)
    }

    pub fn print_trait_item(&mut self, ti: &hir::TraitItem) -> io::Result<()> {
        self.ann.pre(self, NodeSubItem(ti.id))?;
        self.hardbreak_if_not_bol()?;
        self.maybe_print_comment(ti.span.lo)?;
        self.print_outer_attributes(&ti.attrs)?;
        match ti.node {
            hir::TraitItemKind::Const(ref ty, default) => {
                self.print_associated_const(ti.name, &ty, default, &hir::Inherited)?;
            }
            hir::TraitItemKind::Method(ref sig, hir::TraitMethod::Required(ref arg_names)) => {
                self.print_method_sig(ti.name, sig, &hir::Inherited, arg_names, None)?;
                word(&mut self.s, ";")?;
            }
            hir::TraitItemKind::Method(ref sig, hir::TraitMethod::Provided(body)) => {
                self.head("")?;
                self.print_method_sig(ti.name, sig, &hir::Inherited, &[], Some(body))?;
                self.nbsp()?;
                self.end()?; // need to close a box
                self.end()?; // need to close a box
                self.ann.nested(self, Nested::Body(body))?;
            }
            hir::TraitItemKind::Type(ref bounds, ref default) => {
                self.print_associated_type(ti.name,
                                           Some(bounds),
                                           default.as_ref().map(|ty| &**ty))?;
            }
        }
        self.ann.post(self, NodeSubItem(ti.id))
    }

    pub fn print_impl_item(&mut self, ii: &hir::ImplItem) -> io::Result<()> {
        self.ann.pre(self, NodeSubItem(ii.id))?;
        self.hardbreak_if_not_bol()?;
        self.maybe_print_comment(ii.span.lo)?;
        self.print_outer_attributes(&ii.attrs)?;

        match ii.defaultness {
            hir::Defaultness::Default { .. } => self.word_nbsp("default")?,
            hir::Defaultness::Final => (),
        }

        match ii.node {
            hir::ImplItemKind::Const(ref ty, expr) => {
                self.print_associated_const(ii.name, &ty, Some(expr), &ii.vis)?;
            }
            hir::ImplItemKind::Method(ref sig, body) => {
                self.head("")?;
                self.print_method_sig(ii.name, sig, &ii.vis, &[], Some(body))?;
                self.nbsp()?;
                self.end()?; // need to close a box
                self.end()?; // need to close a box
                self.ann.nested(self, Nested::Body(body))?;
            }
            hir::ImplItemKind::Type(ref ty) => {
                self.print_associated_type(ii.name, None, Some(ty))?;
            }
        }
        self.ann.post(self, NodeSubItem(ii.id))
    }

    pub fn print_stmt(&mut self, st: &hir::Stmt) -> io::Result<()> {
        self.maybe_print_comment(st.span.lo)?;
        match st.node {
            hir::StmtDecl(ref decl, _) => {
                self.print_decl(&decl)?;
            }
            hir::StmtExpr(ref expr, _) => {
                self.space_if_not_bol()?;
                self.print_expr(&expr)?;
            }
            hir::StmtSemi(ref expr, _) => {
                self.space_if_not_bol()?;
                self.print_expr(&expr)?;
                word(&mut self.s, ";")?;
            }
        }
        if stmt_ends_with_semi(&st.node) {
            word(&mut self.s, ";")?;
        }
        self.maybe_print_trailing_comment(st.span, None)
    }

    pub fn print_block(&mut self, blk: &hir::Block) -> io::Result<()> {
        self.print_block_with_attrs(blk, &[])
    }

    pub fn print_block_unclosed(&mut self, blk: &hir::Block) -> io::Result<()> {
        self.print_block_unclosed_indent(blk, indent_unit)
    }

    pub fn print_block_unclosed_indent(&mut self,
                                       blk: &hir::Block,
                                       indented: usize)
                                       -> io::Result<()> {
        self.print_block_maybe_unclosed(blk, indented, &[], false)
    }

    pub fn print_block_with_attrs(&mut self,
                                  blk: &hir::Block,
                                  attrs: &[ast::Attribute])
                                  -> io::Result<()> {
        self.print_block_maybe_unclosed(blk, indent_unit, attrs, true)
    }

    pub fn print_block_maybe_unclosed(&mut self,
                                      blk: &hir::Block,
                                      indented: usize,
                                      attrs: &[ast::Attribute],
                                      close_box: bool)
                                      -> io::Result<()> {
        match blk.rules {
            hir::UnsafeBlock(..) => self.word_space("unsafe")?,
            hir::PushUnsafeBlock(..) => self.word_space("push_unsafe")?,
            hir::PopUnsafeBlock(..) => self.word_space("pop_unsafe")?,
            hir::DefaultBlock => (),
        }
        self.maybe_print_comment(blk.span.lo)?;
        self.ann.pre(self, NodeBlock(blk))?;
        self.bopen()?;

        self.print_inner_attributes(attrs)?;

        for st in &blk.stmts {
            self.print_stmt(st)?;
        }
        match blk.expr {
            Some(ref expr) => {
                self.space_if_not_bol()?;
                self.print_expr(&expr)?;
                self.maybe_print_trailing_comment(expr.span, Some(blk.span.hi))?;
            }
            _ => (),
        }
        self.bclose_maybe_open(blk.span, indented, close_box)?;
        self.ann.post(self, NodeBlock(blk))
    }

    fn print_else(&mut self, els: Option<&hir::Expr>) -> io::Result<()> {
        match els {
            Some(_else) => {
                match _else.node {
                    // "another else-if"
                    hir::ExprIf(ref i, ref then, ref e) => {
                        self.cbox(indent_unit - 1)?;
                        self.ibox(0)?;
                        word(&mut self.s, " else if ")?;
                        self.print_expr(&i)?;
                        space(&mut self.s)?;
                        self.print_block(&then)?;
                        self.print_else(e.as_ref().map(|e| &**e))
                    }
                    // "final else"
                    hir::ExprBlock(ref b) => {
                        self.cbox(indent_unit - 1)?;
                        self.ibox(0)?;
                        word(&mut self.s, " else ")?;
                        self.print_block(&b)
                    }
                    // BLEAH, constraints would be great here
                    _ => {
                        panic!("print_if saw if with weird alternative");
                    }
                }
            }
            _ => Ok(()),
        }
    }

    pub fn print_if(&mut self,
                    test: &hir::Expr,
                    blk: &hir::Block,
                    elseopt: Option<&hir::Expr>)
                    -> io::Result<()> {
        self.head("if")?;
        self.print_expr(test)?;
        space(&mut self.s)?;
        self.print_block(blk)?;
        self.print_else(elseopt)
    }

    pub fn print_if_let(&mut self,
                        pat: &hir::Pat,
                        expr: &hir::Expr,
                        blk: &hir::Block,
                        elseopt: Option<&hir::Expr>)
                        -> io::Result<()> {
        self.head("if let")?;
        self.print_pat(pat)?;
        space(&mut self.s)?;
        self.word_space("=")?;
        self.print_expr(expr)?;
        space(&mut self.s)?;
        self.print_block(blk)?;
        self.print_else(elseopt)
    }


    fn print_call_post(&mut self, args: &[hir::Expr]) -> io::Result<()> {
        self.popen()?;
        self.commasep_exprs(Inconsistent, args)?;
        self.pclose()
    }

    pub fn print_expr_maybe_paren(&mut self, expr: &hir::Expr) -> io::Result<()> {
        let needs_par = needs_parentheses(expr);
        if needs_par {
            self.popen()?;
        }
        self.print_expr(expr)?;
        if needs_par {
            self.pclose()?;
        }
        Ok(())
    }

    fn print_expr_vec(&mut self, exprs: &[hir::Expr]) -> io::Result<()> {
        self.ibox(indent_unit)?;
        word(&mut self.s, "[")?;
        self.commasep_exprs(Inconsistent, exprs)?;
        word(&mut self.s, "]")?;
        self.end()
    }

    fn print_expr_repeat(&mut self, element: &hir::Expr, count: hir::BodyId) -> io::Result<()> {
        self.ibox(indent_unit)?;
        word(&mut self.s, "[")?;
        self.print_expr(element)?;
        self.word_space(";")?;
        self.ann.nested(self, Nested::Body(count))?;
        word(&mut self.s, "]")?;
        self.end()
    }

    fn print_expr_struct(&mut self,
                         qpath: &hir::QPath,
                         fields: &[hir::Field],
                         wth: &Option<P<hir::Expr>>)
                         -> io::Result<()> {
        self.print_qpath(qpath, true)?;
        word(&mut self.s, "{")?;
        self.commasep_cmnt(Consistent,
                           &fields[..],
                           |s, field| {
                               s.ibox(indent_unit)?;
                               if !field.is_shorthand {
                                    s.print_name(field.name.node)?;
                                    s.word_space(":")?;
                               }
                               s.print_expr(&field.expr)?;
                               s.end()
                           },
                           |f| f.span)?;
        match *wth {
            Some(ref expr) => {
                self.ibox(indent_unit)?;
                if !fields.is_empty() {
                    word(&mut self.s, ",")?;
                    space(&mut self.s)?;
                }
                word(&mut self.s, "..")?;
                self.print_expr(&expr)?;
                self.end()?;
            }
            _ => if !fields.is_empty() {
                word(&mut self.s, ",")?
            },
        }
        word(&mut self.s, "}")?;
        Ok(())
    }

    fn print_expr_tup(&mut self, exprs: &[hir::Expr]) -> io::Result<()> {
        self.popen()?;
        self.commasep_exprs(Inconsistent, exprs)?;
        if exprs.len() == 1 {
            word(&mut self.s, ",")?;
        }
        self.pclose()
    }

    fn print_expr_call(&mut self, func: &hir::Expr, args: &[hir::Expr]) -> io::Result<()> {
        self.print_expr_maybe_paren(func)?;
        self.print_call_post(args)
    }

    fn print_expr_method_call(&mut self,
                              name: Spanned<ast::Name>,
                              tys: &[P<hir::Ty>],
                              args: &[hir::Expr])
                              -> io::Result<()> {
        let base_args = &args[1..];
        self.print_expr(&args[0])?;
        word(&mut self.s, ".")?;
        self.print_name(name.node)?;
        if !tys.is_empty() {
            word(&mut self.s, "::<")?;
            self.commasep(Inconsistent, tys, |s, ty| s.print_type(&ty))?;
            word(&mut self.s, ">")?;
        }
        self.print_call_post(base_args)
    }

    fn print_expr_binary(&mut self,
                         op: hir::BinOp,
                         lhs: &hir::Expr,
                         rhs: &hir::Expr)
                         -> io::Result<()> {
        self.print_expr(lhs)?;
        space(&mut self.s)?;
        self.word_space(op.node.as_str())?;
        self.print_expr(rhs)
    }

    fn print_expr_unary(&mut self, op: hir::UnOp, expr: &hir::Expr) -> io::Result<()> {
        word(&mut self.s, op.as_str())?;
        self.print_expr_maybe_paren(expr)
    }

    fn print_expr_addr_of(&mut self,
                          mutability: hir::Mutability,
                          expr: &hir::Expr)
                          -> io::Result<()> {
        word(&mut self.s, "&")?;
        self.print_mutability(mutability)?;
        self.print_expr_maybe_paren(expr)
    }

    pub fn print_expr(&mut self, expr: &hir::Expr) -> io::Result<()> {
        self.maybe_print_comment(expr.span.lo)?;
        self.print_outer_attributes(&expr.attrs)?;
        self.ibox(indent_unit)?;
        self.ann.pre(self, NodeExpr(expr))?;
        match expr.node {
            hir::ExprBox(ref expr) => {
                self.word_space("box")?;
                self.print_expr(expr)?;
            }
            hir::ExprArray(ref exprs) => {
                self.print_expr_vec(exprs)?;
            }
            hir::ExprRepeat(ref element, count) => {
                self.print_expr_repeat(&element, count)?;
            }
            hir::ExprStruct(ref qpath, ref fields, ref wth) => {
                self.print_expr_struct(qpath, &fields[..], wth)?;
            }
            hir::ExprTup(ref exprs) => {
                self.print_expr_tup(exprs)?;
            }
            hir::ExprCall(ref func, ref args) => {
                self.print_expr_call(&func, args)?;
            }
            hir::ExprMethodCall(name, ref tys, ref args) => {
                self.print_expr_method_call(name, &tys[..], args)?;
            }
            hir::ExprBinary(op, ref lhs, ref rhs) => {
                self.print_expr_binary(op, &lhs, &rhs)?;
            }
            hir::ExprUnary(op, ref expr) => {
                self.print_expr_unary(op, &expr)?;
            }
            hir::ExprAddrOf(m, ref expr) => {
                self.print_expr_addr_of(m, &expr)?;
            }
            hir::ExprLit(ref lit) => {
                self.print_literal(&lit)?;
            }
            hir::ExprCast(ref expr, ref ty) => {
                self.print_expr(&expr)?;
                space(&mut self.s)?;
                self.word_space("as")?;
                self.print_type(&ty)?;
            }
            hir::ExprType(ref expr, ref ty) => {
                self.print_expr(&expr)?;
                self.word_space(":")?;
                self.print_type(&ty)?;
            }
            hir::ExprIf(ref test, ref blk, ref elseopt) => {
                self.print_if(&test, &blk, elseopt.as_ref().map(|e| &**e))?;
            }
            hir::ExprWhile(ref test, ref blk, opt_sp_name) => {
                if let Some(sp_name) = opt_sp_name {
                    self.print_name(sp_name.node)?;
                    self.word_space(":")?;
                }
                self.head("while")?;
                self.print_expr(&test)?;
                space(&mut self.s)?;
                self.print_block(&blk)?;
            }
            hir::ExprLoop(ref blk, opt_sp_name, _) => {
                if let Some(sp_name) = opt_sp_name {
                    self.print_name(sp_name.node)?;
                    self.word_space(":")?;
                }
                self.head("loop")?;
                space(&mut self.s)?;
                self.print_block(&blk)?;
            }
            hir::ExprMatch(ref expr, ref arms, _) => {
                self.cbox(indent_unit)?;
                self.ibox(4)?;
                self.word_nbsp("match")?;
                self.print_expr(&expr)?;
                space(&mut self.s)?;
                self.bopen()?;
                for arm in arms {
                    self.print_arm(arm)?;
                }
                self.bclose_(expr.span, indent_unit)?;
            }
            hir::ExprClosure(capture_clause, ref decl, body, _fn_decl_span) => {
                self.print_capture_clause(capture_clause)?;

                self.print_closure_args(&decl, body)?;
                space(&mut self.s)?;

                // this is a bare expression
                self.ann.nested(self, Nested::Body(body))?;
                self.end()?; // need to close a box

                // a box will be closed by print_expr, but we didn't want an overall
                // wrapper so we closed the corresponding opening. so create an
                // empty box to satisfy the close.
                self.ibox(0)?;
            }
            hir::ExprBlock(ref blk) => {
                // containing cbox, will be closed by print-block at }
                self.cbox(indent_unit)?;
                // head-box, will be closed by print-block after {
                self.ibox(0)?;
                self.print_block(&blk)?;
            }
            hir::ExprAssign(ref lhs, ref rhs) => {
                self.print_expr(&lhs)?;
                space(&mut self.s)?;
                self.word_space("=")?;
                self.print_expr(&rhs)?;
            }
            hir::ExprAssignOp(op, ref lhs, ref rhs) => {
                self.print_expr(&lhs)?;
                space(&mut self.s)?;
                word(&mut self.s, op.node.as_str())?;
                self.word_space("=")?;
                self.print_expr(&rhs)?;
            }
            hir::ExprField(ref expr, name) => {
                self.print_expr(&expr)?;
                word(&mut self.s, ".")?;
                self.print_name(name.node)?;
            }
            hir::ExprTupField(ref expr, id) => {
                self.print_expr(&expr)?;
                word(&mut self.s, ".")?;
                self.print_usize(id.node)?;
            }
            hir::ExprIndex(ref expr, ref index) => {
                self.print_expr(&expr)?;
                word(&mut self.s, "[")?;
                self.print_expr(&index)?;
                word(&mut self.s, "]")?;
            }
            hir::ExprPath(ref qpath) => {
                self.print_qpath(qpath, true)?
            }
            hir::ExprBreak(opt_label, ref opt_expr) => {
                word(&mut self.s, "break")?;
                space(&mut self.s)?;
                if let Some(label) = opt_label {
                    self.print_name(label.name)?;
                    space(&mut self.s)?;
                }
                if let Some(ref expr) = *opt_expr {
                    self.print_expr(expr)?;
                    space(&mut self.s)?;
                }
            }
            hir::ExprAgain(opt_label) => {
                word(&mut self.s, "continue")?;
                space(&mut self.s)?;
                if let Some(label) = opt_label {
                    self.print_name(label.name)?;
                    space(&mut self.s)?
                }
            }
            hir::ExprRet(ref result) => {
                word(&mut self.s, "return")?;
                match *result {
                    Some(ref expr) => {
                        word(&mut self.s, " ")?;
                        self.print_expr(&expr)?;
                    }
                    _ => (),
                }
            }
            hir::ExprInlineAsm(ref a, ref outputs, ref inputs) => {
                word(&mut self.s, "asm!")?;
                self.popen()?;
                self.print_string(&a.asm.as_str(), a.asm_str_style)?;
                self.word_space(":")?;

                let mut out_idx = 0;
                self.commasep(Inconsistent, &a.outputs, |s, out| {
                    let constraint = out.constraint.as_str();
                    let mut ch = constraint.chars();
                    match ch.next() {
                        Some('=') if out.is_rw => {
                            s.print_string(&format!("+{}", ch.as_str()),
                                           ast::StrStyle::Cooked)?
                        }
                        _ => s.print_string(&constraint, ast::StrStyle::Cooked)?,
                    }
                    s.popen()?;
                    s.print_expr(&outputs[out_idx])?;
                    s.pclose()?;
                    out_idx += 1;
                    Ok(())
                })?;
                space(&mut self.s)?;
                self.word_space(":")?;

                let mut in_idx = 0;
                self.commasep(Inconsistent, &a.inputs, |s, co| {
                    s.print_string(&co.as_str(), ast::StrStyle::Cooked)?;
                    s.popen()?;
                    s.print_expr(&inputs[in_idx])?;
                    s.pclose()?;
                    in_idx += 1;
                    Ok(())
                })?;
                space(&mut self.s)?;
                self.word_space(":")?;

                self.commasep(Inconsistent, &a.clobbers, |s, co| {
                    s.print_string(&co.as_str(), ast::StrStyle::Cooked)?;
                    Ok(())
                })?;

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
                    space(&mut self.s)?;
                    self.word_space(":")?;
                    self.commasep(Inconsistent, &options, |s, &co| {
                        s.print_string(co, ast::StrStyle::Cooked)?;
                        Ok(())
                    })?;
                }

                self.pclose()?;
            }
        }
        self.ann.post(self, NodeExpr(expr))?;
        self.end()
    }

    pub fn print_local_decl(&mut self, loc: &hir::Local) -> io::Result<()> {
        self.print_pat(&loc.pat)?;
        if let Some(ref ty) = loc.ty {
            self.word_space(":")?;
            self.print_type(&ty)?;
        }
        Ok(())
    }

    pub fn print_decl(&mut self, decl: &hir::Decl) -> io::Result<()> {
        self.maybe_print_comment(decl.span.lo)?;
        match decl.node {
            hir::DeclLocal(ref loc) => {
                self.space_if_not_bol()?;
                self.ibox(indent_unit)?;
                self.word_nbsp("let")?;

                self.ibox(indent_unit)?;
                self.print_local_decl(&loc)?;
                self.end()?;
                if let Some(ref init) = loc.init {
                    self.nbsp()?;
                    self.word_space("=")?;
                    self.print_expr(&init)?;
                }
                self.end()
            }
            hir::DeclItem(item) => {
                self.ann.nested(self, Nested::Item(item))
            }
        }
    }

    pub fn print_usize(&mut self, i: usize) -> io::Result<()> {
        word(&mut self.s, &i.to_string())
    }

    pub fn print_name(&mut self, name: ast::Name) -> io::Result<()> {
        word(&mut self.s, &name.as_str())?;
        self.ann.post(self, NodeName(&name))
    }

    pub fn print_for_decl(&mut self, loc: &hir::Local, coll: &hir::Expr) -> io::Result<()> {
        self.print_local_decl(loc)?;
        space(&mut self.s)?;
        self.word_space("in")?;
        self.print_expr(coll)
    }

    pub fn print_path(&mut self,
                      path: &hir::Path,
                      colons_before_params: bool)
                      -> io::Result<()> {
        self.maybe_print_comment(path.span.lo)?;

        for (i, segment) in path.segments.iter().enumerate() {
            if i > 0 {
                word(&mut self.s, "::")?
            }
            if segment.name != keywords::CrateRoot.name() && segment.name != "$crate" {
                self.print_name(segment.name)?;
                self.print_path_parameters(&segment.parameters, colons_before_params)?;
            }
        }

        Ok(())
    }

    pub fn print_qpath(&mut self,
                       qpath: &hir::QPath,
                       colons_before_params: bool)
                       -> io::Result<()> {
        match *qpath {
            hir::QPath::Resolved(None, ref path) => {
                self.print_path(path, colons_before_params)
            }
            hir::QPath::Resolved(Some(ref qself), ref path) => {
                word(&mut self.s, "<")?;
                self.print_type(qself)?;
                space(&mut self.s)?;
                self.word_space("as")?;

                for (i, segment) in path.segments[..path.segments.len() - 1].iter().enumerate() {
                    if i > 0 {
                        word(&mut self.s, "::")?
                    }
                    if segment.name != keywords::CrateRoot.name() && segment.name != "$crate" {
                        self.print_name(segment.name)?;
                        self.print_path_parameters(&segment.parameters, colons_before_params)?;
                    }
                }

                word(&mut self.s, ">")?;
                word(&mut self.s, "::")?;
                let item_segment = path.segments.last().unwrap();
                self.print_name(item_segment.name)?;
                self.print_path_parameters(&item_segment.parameters, colons_before_params)
            }
            hir::QPath::TypeRelative(ref qself, ref item_segment) => {
                word(&mut self.s, "<")?;
                self.print_type(qself)?;
                word(&mut self.s, ">")?;
                word(&mut self.s, "::")?;
                self.print_name(item_segment.name)?;
                self.print_path_parameters(&item_segment.parameters, colons_before_params)
            }
        }
    }

    fn print_path_parameters(&mut self,
                             parameters: &hir::PathParameters,
                             colons_before_params: bool)
                             -> io::Result<()> {
        if parameters.is_empty() {
            let infer_types = match *parameters {
                hir::AngleBracketedParameters(ref data) => data.infer_types,
                hir::ParenthesizedParameters(_) => false
            };

            // FIXME(eddyb) See the comment below about infer_types.
            if !(infer_types && false) {
                return Ok(());
            }
        }

        if colons_before_params {
            word(&mut self.s, "::")?
        }

        match *parameters {
            hir::AngleBracketedParameters(ref data) => {
                word(&mut self.s, "<")?;

                let mut comma = false;
                for lifetime in &data.lifetimes {
                    if comma {
                        self.word_space(",")?
                    }
                    self.print_lifetime(lifetime)?;
                    comma = true;
                }

                if !data.types.is_empty() {
                    if comma {
                        self.word_space(",")?
                    }
                    self.commasep(Inconsistent, &data.types, |s, ty| s.print_type(&ty))?;
                    comma = true;
                }

                // FIXME(eddyb) This would leak into error messages, e.g.:
                // "non-exhaustive patterns: `Some::<..>(_)` not covered".
                if data.infer_types && false {
                    if comma {
                        self.word_space(",")?
                    }
                    word(&mut self.s, "..")?;
                    comma = true;
                }

                for binding in data.bindings.iter() {
                    if comma {
                        self.word_space(",")?
                    }
                    self.print_name(binding.name)?;
                    space(&mut self.s)?;
                    self.word_space("=")?;
                    self.print_type(&binding.ty)?;
                    comma = true;
                }

                word(&mut self.s, ">")?
            }

            hir::ParenthesizedParameters(ref data) => {
                word(&mut self.s, "(")?;
                self.commasep(Inconsistent, &data.inputs, |s, ty| s.print_type(&ty))?;
                word(&mut self.s, ")")?;

                if let Some(ref ty) = data.output {
                    self.space_if_not_bol()?;
                    self.word_space("->")?;
                    self.print_type(&ty)?;
                }
            }
        }

        Ok(())
    }

    pub fn print_pat(&mut self, pat: &hir::Pat) -> io::Result<()> {
        self.maybe_print_comment(pat.span.lo)?;
        self.ann.pre(self, NodePat(pat))?;
        // Pat isn't normalized, but the beauty of it
        // is that it doesn't matter
        match pat.node {
            PatKind::Wild => word(&mut self.s, "_")?,
            PatKind::Binding(binding_mode, _, ref path1, ref sub) => {
                match binding_mode {
                    hir::BindByRef(mutbl) => {
                        self.word_nbsp("ref")?;
                        self.print_mutability(mutbl)?;
                    }
                    hir::BindByValue(hir::MutImmutable) => {}
                    hir::BindByValue(hir::MutMutable) => {
                        self.word_nbsp("mut")?;
                    }
                }
                self.print_name(path1.node)?;
                if let Some(ref p) = *sub {
                    word(&mut self.s, "@")?;
                    self.print_pat(&p)?;
                }
            }
            PatKind::TupleStruct(ref qpath, ref elts, ddpos) => {
                self.print_qpath(qpath, true)?;
                self.popen()?;
                if let Some(ddpos) = ddpos {
                    self.commasep(Inconsistent, &elts[..ddpos], |s, p| s.print_pat(&p))?;
                    if ddpos != 0 {
                        self.word_space(",")?;
                    }
                    word(&mut self.s, "..")?;
                    if ddpos != elts.len() {
                        word(&mut self.s, ",")?;
                        self.commasep(Inconsistent, &elts[ddpos..], |s, p| s.print_pat(&p))?;
                    }
                } else {
                    self.commasep(Inconsistent, &elts[..], |s, p| s.print_pat(&p))?;
                }
                self.pclose()?;
            }
            PatKind::Path(ref qpath) => {
                self.print_qpath(qpath, true)?;
            }
            PatKind::Struct(ref qpath, ref fields, etc) => {
                self.print_qpath(qpath, true)?;
                self.nbsp()?;
                self.word_space("{")?;
                self.commasep_cmnt(Consistent,
                                   &fields[..],
                                   |s, f| {
                                       s.cbox(indent_unit)?;
                                       if !f.node.is_shorthand {
                                           s.print_name(f.node.name)?;
                                           s.word_nbsp(":")?;
                                       }
                                       s.print_pat(&f.node.pat)?;
                                       s.end()
                                   },
                                   |f| f.node.pat.span)?;
                if etc {
                    if !fields.is_empty() {
                        self.word_space(",")?;
                    }
                    word(&mut self.s, "..")?;
                }
                space(&mut self.s)?;
                word(&mut self.s, "}")?;
            }
            PatKind::Tuple(ref elts, ddpos) => {
                self.popen()?;
                if let Some(ddpos) = ddpos {
                    self.commasep(Inconsistent, &elts[..ddpos], |s, p| s.print_pat(&p))?;
                    if ddpos != 0 {
                        self.word_space(",")?;
                    }
                    word(&mut self.s, "..")?;
                    if ddpos != elts.len() {
                        word(&mut self.s, ",")?;
                        self.commasep(Inconsistent, &elts[ddpos..], |s, p| s.print_pat(&p))?;
                    }
                } else {
                    self.commasep(Inconsistent, &elts[..], |s, p| s.print_pat(&p))?;
                    if elts.len() == 1 {
                        word(&mut self.s, ",")?;
                    }
                }
                self.pclose()?;
            }
            PatKind::Box(ref inner) => {
                word(&mut self.s, "box ")?;
                self.print_pat(&inner)?;
            }
            PatKind::Ref(ref inner, mutbl) => {
                word(&mut self.s, "&")?;
                if mutbl == hir::MutMutable {
                    word(&mut self.s, "mut ")?;
                }
                self.print_pat(&inner)?;
            }
            PatKind::Lit(ref e) => self.print_expr(&e)?,
            PatKind::Range(ref begin, ref end) => {
                self.print_expr(&begin)?;
                space(&mut self.s)?;
                word(&mut self.s, "...")?;
                self.print_expr(&end)?;
            }
            PatKind::Slice(ref before, ref slice, ref after) => {
                word(&mut self.s, "[")?;
                self.commasep(Inconsistent, &before[..], |s, p| s.print_pat(&p))?;
                if let Some(ref p) = *slice {
                    if !before.is_empty() {
                        self.word_space(",")?;
                    }
                    if p.node != PatKind::Wild {
                        self.print_pat(&p)?;
                    }
                    word(&mut self.s, "..")?;
                    if !after.is_empty() {
                        self.word_space(",")?;
                    }
                }
                self.commasep(Inconsistent, &after[..], |s, p| s.print_pat(&p))?;
                word(&mut self.s, "]")?;
            }
        }
        self.ann.post(self, NodePat(pat))
    }

    fn print_arm(&mut self, arm: &hir::Arm) -> io::Result<()> {
        // I have no idea why this check is necessary, but here it
        // is :(
        if arm.attrs.is_empty() {
            space(&mut self.s)?;
        }
        self.cbox(indent_unit)?;
        self.ibox(0)?;
        self.print_outer_attributes(&arm.attrs)?;
        let mut first = true;
        for p in &arm.pats {
            if first {
                first = false;
            } else {
                space(&mut self.s)?;
                self.word_space("|")?;
            }
            self.print_pat(&p)?;
        }
        space(&mut self.s)?;
        if let Some(ref e) = arm.guard {
            self.word_space("if")?;
            self.print_expr(&e)?;
            space(&mut self.s)?;
        }
        self.word_space("=>")?;

        match arm.body.node {
            hir::ExprBlock(ref blk) => {
                // the block will close the pattern's ibox
                self.print_block_unclosed_indent(&blk, indent_unit)?;

                // If it is a user-provided unsafe block, print a comma after it
                if let hir::UnsafeBlock(hir::UserProvided) = blk.rules {
                    word(&mut self.s, ",")?;
                }
            }
            _ => {
                self.end()?; // close the ibox for the pattern
                self.print_expr(&arm.body)?;
                word(&mut self.s, ",")?;
            }
        }
        self.end() // close enclosing cbox
    }

    pub fn print_fn(&mut self,
                    decl: &hir::FnDecl,
                    unsafety: hir::Unsafety,
                    constness: hir::Constness,
                    abi: Abi,
                    name: Option<ast::Name>,
                    generics: &hir::Generics,
                    vis: &hir::Visibility,
                    arg_names: &[Spanned<ast::Name>],
                    body_id: Option<hir::BodyId>)
                    -> io::Result<()> {
        self.print_fn_header_info(unsafety, constness, abi, vis)?;

        if let Some(name) = name {
            self.nbsp()?;
            self.print_name(name)?;
        }
        self.print_generics(generics)?;

        self.popen()?;
        let mut i = 0;
        // Make sure we aren't supplied *both* `arg_names` and `body_id`.
        assert!(arg_names.is_empty() || body_id.is_none());
        self.commasep(Inconsistent, &decl.inputs, |s, ty| {
            s.ibox(indent_unit)?;
            if let Some(name) = arg_names.get(i) {
                word(&mut s.s, &name.node.as_str())?;
                word(&mut s.s, ":")?;
                space(&mut s.s)?;
            } else if let Some(body_id) = body_id {
                s.ann.nested(s, Nested::BodyArgPat(body_id, i))?;
                word(&mut s.s, ":")?;
                space(&mut s.s)?;
            }
            i += 1;
            s.print_type(ty)?;
            s.end()
        })?;
        if decl.variadic {
            word(&mut self.s, ", ...")?;
        }
        self.pclose()?;

        self.print_fn_output(decl)?;
        self.print_where_clause(&generics.where_clause)
    }

    fn print_closure_args(&mut self, decl: &hir::FnDecl, body_id: hir::BodyId) -> io::Result<()> {
        word(&mut self.s, "|")?;
        let mut i = 0;
        self.commasep(Inconsistent, &decl.inputs, |s, ty| {
            s.ibox(indent_unit)?;

            s.ann.nested(s, Nested::BodyArgPat(body_id, i))?;
            i += 1;

            if ty.node != hir::TyInfer {
                word(&mut s.s, ":")?;
                space(&mut s.s)?;
                s.print_type(ty)?;
            }
            s.end()
        })?;
        word(&mut self.s, "|")?;

        if let hir::DefaultReturn(..) = decl.output {
            return Ok(());
        }

        self.space_if_not_bol()?;
        self.word_space("->")?;
        match decl.output {
            hir::Return(ref ty) => {
                self.print_type(&ty)?;
                self.maybe_print_comment(ty.span.lo)
            }
            hir::DefaultReturn(..) => unreachable!(),
        }
    }

    pub fn print_capture_clause(&mut self, capture_clause: hir::CaptureClause) -> io::Result<()> {
        match capture_clause {
            hir::CaptureByValue => self.word_space("move"),
            hir::CaptureByRef => Ok(()),
        }
    }

    pub fn print_bounds(&mut self, prefix: &str, bounds: &[hir::TyParamBound]) -> io::Result<()> {
        if !bounds.is_empty() {
            word(&mut self.s, prefix)?;
            let mut first = true;
            for bound in bounds {
                self.nbsp()?;
                if first {
                    first = false;
                } else {
                    self.word_space("+")?;
                }

                match *bound {
                    TraitTyParamBound(ref tref, TraitBoundModifier::None) => {
                        self.print_poly_trait_ref(tref)
                    }
                    TraitTyParamBound(ref tref, TraitBoundModifier::Maybe) => {
                        word(&mut self.s, "?")?;
                        self.print_poly_trait_ref(tref)
                    }
                    RegionTyParamBound(ref lt) => {
                        self.print_lifetime(lt)
                    }
                }?
            }
            Ok(())
        } else {
            Ok(())
        }
    }

    pub fn print_lifetime(&mut self, lifetime: &hir::Lifetime) -> io::Result<()> {
        self.print_name(lifetime.name)
    }

    pub fn print_lifetime_def(&mut self, lifetime: &hir::LifetimeDef) -> io::Result<()> {
        self.print_lifetime(&lifetime.lifetime)?;
        let mut sep = ":";
        for v in &lifetime.bounds {
            word(&mut self.s, sep)?;
            self.print_lifetime(v)?;
            sep = "+";
        }
        Ok(())
    }

    pub fn print_generics(&mut self, generics: &hir::Generics) -> io::Result<()> {
        let total = generics.lifetimes.len() + generics.ty_params.len();
        if total == 0 {
            return Ok(());
        }

        word(&mut self.s, "<")?;

        let mut ints = Vec::new();
        for i in 0..total {
            ints.push(i);
        }

        self.commasep(Inconsistent, &ints[..], |s, &idx| {
            if idx < generics.lifetimes.len() {
                let lifetime = &generics.lifetimes[idx];
                s.print_lifetime_def(lifetime)
            } else {
                let idx = idx - generics.lifetimes.len();
                let param = &generics.ty_params[idx];
                s.print_ty_param(param)
            }
        })?;

        word(&mut self.s, ">")?;
        Ok(())
    }

    pub fn print_ty_param(&mut self, param: &hir::TyParam) -> io::Result<()> {
        self.print_name(param.name)?;
        self.print_bounds(":", &param.bounds)?;
        match param.default {
            Some(ref default) => {
                space(&mut self.s)?;
                self.word_space("=")?;
                self.print_type(&default)
            }
            _ => Ok(()),
        }
    }

    pub fn print_where_clause(&mut self, where_clause: &hir::WhereClause) -> io::Result<()> {
        if where_clause.predicates.is_empty() {
            return Ok(());
        }

        space(&mut self.s)?;
        self.word_space("where")?;

        for (i, predicate) in where_clause.predicates.iter().enumerate() {
            if i != 0 {
                self.word_space(",")?;
            }

            match predicate {
                &hir::WherePredicate::BoundPredicate(hir::WhereBoundPredicate{ref bound_lifetimes,
                                                                              ref bounded_ty,
                                                                              ref bounds,
                                                                              ..}) => {
                    self.print_formal_lifetime_list(bound_lifetimes)?;
                    self.print_type(&bounded_ty)?;
                    self.print_bounds(":", bounds)?;
                }
                &hir::WherePredicate::RegionPredicate(hir::WhereRegionPredicate{ref lifetime,
                                                                                ref bounds,
                                                                                ..}) => {
                    self.print_lifetime(lifetime)?;
                    word(&mut self.s, ":")?;

                    for (i, bound) in bounds.iter().enumerate() {
                        self.print_lifetime(bound)?;

                        if i != 0 {
                            word(&mut self.s, ":")?;
                        }
                    }
                }
                &hir::WherePredicate::EqPredicate(hir::WhereEqPredicate{ref lhs_ty,
                                                                        ref rhs_ty,
                                                                        ..}) => {
                    self.print_type(lhs_ty)?;
                    space(&mut self.s)?;
                    self.word_space("=")?;
                    self.print_type(rhs_ty)?;
                }
            }
        }

        Ok(())
    }

    pub fn print_mutability(&mut self, mutbl: hir::Mutability) -> io::Result<()> {
        match mutbl {
            hir::MutMutable => self.word_nbsp("mut"),
            hir::MutImmutable => Ok(()),
        }
    }

    pub fn print_mt(&mut self, mt: &hir::MutTy) -> io::Result<()> {
        self.print_mutability(mt.mutbl)?;
        self.print_type(&mt.ty)
    }

    pub fn print_fn_output(&mut self, decl: &hir::FnDecl) -> io::Result<()> {
        if let hir::DefaultReturn(..) = decl.output {
            return Ok(());
        }

        self.space_if_not_bol()?;
        self.ibox(indent_unit)?;
        self.word_space("->")?;
        match decl.output {
            hir::DefaultReturn(..) => unreachable!(),
            hir::Return(ref ty) => self.print_type(&ty)?,
        }
        self.end()?;

        match decl.output {
            hir::Return(ref output) => self.maybe_print_comment(output.span.lo),
            _ => Ok(()),
        }
    }

    pub fn print_ty_fn(&mut self,
                       abi: Abi,
                       unsafety: hir::Unsafety,
                       decl: &hir::FnDecl,
                       name: Option<ast::Name>,
                       generics: &hir::Generics)
                       -> io::Result<()> {
        self.ibox(indent_unit)?;
        if !generics.lifetimes.is_empty() || !generics.ty_params.is_empty() {
            word(&mut self.s, "for")?;
            self.print_generics(generics)?;
        }
        let generics = hir::Generics {
            lifetimes: hir::HirVec::new(),
            ty_params: hir::HirVec::new(),
            where_clause: hir::WhereClause {
                id: ast::DUMMY_NODE_ID,
                predicates: hir::HirVec::new(),
            },
            span: syntax_pos::DUMMY_SP,
        };
        self.print_fn(decl,
                      unsafety,
                      hir::Constness::NotConst,
                      abi,
                      name,
                      &generics,
                      &hir::Inherited,
                      &[],
                      None)?;
        self.end()
    }

    pub fn maybe_print_trailing_comment(&mut self,
                                        span: syntax_pos::Span,
                                        next_pos: Option<BytePos>)
                                        -> io::Result<()> {
        let cm = match self.cm {
            Some(cm) => cm,
            _ => return Ok(()),
        };
        if let Some(ref cmnt) = self.next_comment() {
            if (*cmnt).style != comments::Trailing {
                return Ok(());
            }
            let span_line = cm.lookup_char_pos(span.hi);
            let comment_line = cm.lookup_char_pos((*cmnt).pos);
            let mut next = (*cmnt).pos + BytePos(1);
            if let Some(p) = next_pos {
                next = p;
            }
            if span.hi < (*cmnt).pos && (*cmnt).pos < next &&
               span_line.line == comment_line.line {
                self.print_comment(cmnt)?;
                self.cur_cmnt_and_lit.cur_cmnt += 1;
            }
        }
        Ok(())
    }

    pub fn print_remaining_comments(&mut self) -> io::Result<()> {
        // If there aren't any remaining comments, then we need to manually
        // make sure there is a line break at the end.
        if self.next_comment().is_none() {
            hardbreak(&mut self.s)?;
        }
        loop {
            match self.next_comment() {
                Some(ref cmnt) => {
                    self.print_comment(cmnt)?;
                    self.cur_cmnt_and_lit.cur_cmnt += 1;
                }
                _ => break,
            }
        }
        Ok(())
    }

    pub fn print_opt_abi_and_extern_if_nondefault(&mut self,
                                                  opt_abi: Option<Abi>)
                                                  -> io::Result<()> {
        match opt_abi {
            Some(Abi::Rust) => Ok(()),
            Some(abi) => {
                self.word_nbsp("extern")?;
                self.word_nbsp(&abi.to_string())
            }
            None => Ok(()),
        }
    }

    pub fn print_extern_opt_abi(&mut self, opt_abi: Option<Abi>) -> io::Result<()> {
        match opt_abi {
            Some(abi) => {
                self.word_nbsp("extern")?;
                self.word_nbsp(&abi.to_string())
            }
            None => Ok(()),
        }
    }

    pub fn print_fn_header_info(&mut self,
                                unsafety: hir::Unsafety,
                                constness: hir::Constness,
                                abi: Abi,
                                vis: &hir::Visibility)
                                -> io::Result<()> {
        word(&mut self.s, &visibility_qualified(vis, ""))?;
        self.print_unsafety(unsafety)?;

        match constness {
            hir::Constness::NotConst => {}
            hir::Constness::Const => self.word_nbsp("const")?,
        }

        if abi != Abi::Rust {
            self.word_nbsp("extern")?;
            self.word_nbsp(&abi.to_string())?;
        }

        word(&mut self.s, "fn")
    }

    pub fn print_unsafety(&mut self, s: hir::Unsafety) -> io::Result<()> {
        match s {
            hir::Unsafety::Normal => Ok(()),
            hir::Unsafety::Unsafe => self.word_nbsp("unsafe"),
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
        hir::ExprIf(..) |
        hir::ExprMatch(..) |
        hir::ExprBlock(_) |
        hir::ExprWhile(..) |
        hir::ExprLoop(..) => false,
        _ => true,
    }
}

/// this statement requires a semicolon after it.
/// note that in one case (stmt_semi), we've already
/// seen the semicolon, and thus don't need another.
fn stmt_ends_with_semi(stmt: &hir::Stmt_) -> bool {
    match *stmt {
        hir::StmtDecl(ref d, _) => {
            match d.node {
                hir::DeclLocal(_) => true,
                hir::DeclItem(_) => false,
            }
        }
        hir::StmtExpr(ref e, _) => {
            expr_requires_semi_to_be_stmt(&e)
        }
        hir::StmtSemi(..) => {
            false
        }
    }
}
