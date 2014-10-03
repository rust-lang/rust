// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi;
use ast::{FnUnboxedClosureKind, FnMutUnboxedClosureKind};
use ast::{FnOnceUnboxedClosureKind};
use ast::{MethodImplItem, RegionTyParamBound, TraitTyParamBound};
use ast::{RequiredMethod, ProvidedMethod, TypeImplItem, TypeTraitItem};
use ast::{UnboxedClosureKind, UnboxedFnTyParamBound};
use ast;
use ast_util;
use owned_slice::OwnedSlice;
use attr::{AttrMetaMethods, AttributeMethods};
use codemap::{CodeMap, BytePos};
use codemap;
use diagnostic;
use parse::token;
use parse::lexer::comments;
use parse;
use print::pp::{break_offset, word, space, zerobreak, hardbreak};
use print::pp::{Breaks, Consistent, Inconsistent, eof};
use print::pp;
use ptr::P;

use std::io::{IoResult, MemWriter};
use std::io;
use std::mem;

pub enum AnnNode<'a> {
    NodeIdent(&'a ast::Ident),
    NodeName(&'a ast::Name),
    NodeBlock(&'a ast::Block),
    NodeItem(&'a ast::Item),
    NodeExpr(&'a ast::Expr),
    NodePat(&'a ast::Pat),
}

pub trait PpAnn {
    fn pre(&self, _state: &mut State, _node: AnnNode) -> IoResult<()> { Ok(()) }
    fn post(&self, _state: &mut State, _node: AnnNode) -> IoResult<()> { Ok(()) }
}

pub struct NoAnn;

impl PpAnn for NoAnn {}

pub struct CurrentCommentAndLiteral {
    cur_cmnt: uint,
    cur_lit: uint,
}

pub struct State<'a> {
    pub s: pp::Printer,
    cm: Option<&'a CodeMap>,
    comments: Option<Vec<comments::Comment> >,
    literals: Option<Vec<comments::Literal> >,
    cur_cmnt_and_lit: CurrentCommentAndLiteral,
    boxes: Vec<pp::Breaks>,
    ann: &'a PpAnn+'a,
    encode_idents_with_hygiene: bool,
}

pub fn rust_printer(writer: Box<io::Writer+'static>) -> State<'static> {
    static NO_ANN: NoAnn = NoAnn;
    rust_printer_annotated(writer, &NO_ANN)
}

pub fn rust_printer_annotated<'a>(writer: Box<io::Writer+'static>,
                                  ann: &'a PpAnn) -> State<'a> {
    State {
        s: pp::mk_printer(writer, default_columns),
        cm: None,
        comments: None,
        literals: None,
        cur_cmnt_and_lit: CurrentCommentAndLiteral {
            cur_cmnt: 0,
            cur_lit: 0
        },
        boxes: Vec::new(),
        ann: ann,
        encode_idents_with_hygiene: false,
    }
}

#[allow(non_uppercase_statics)]
pub static indent_unit: uint = 4u;

#[allow(non_uppercase_statics)]
pub static default_columns: uint = 78u;

/// Requires you to pass an input filename and reader so that
/// it can scan the input text for comments and literals to
/// copy forward.
pub fn print_crate<'a>(cm: &'a CodeMap,
                       span_diagnostic: &diagnostic::SpanHandler,
                       krate: &ast::Crate,
                       filename: String,
                       input: &mut io::Reader,
                       out: Box<io::Writer+'static>,
                       ann: &'a PpAnn,
                       is_expanded: bool) -> IoResult<()> {
    let mut s = State::new_from_input(cm,
                                      span_diagnostic,
                                      filename,
                                      input,
                                      out,
                                      ann,
                                      is_expanded);
    try!(s.print_mod(&krate.module, krate.attrs.as_slice()));
    try!(s.print_remaining_comments());
    eof(&mut s.s)
}

impl<'a> State<'a> {
    pub fn new_from_input(cm: &'a CodeMap,
                          span_diagnostic: &diagnostic::SpanHandler,
                          filename: String,
                          input: &mut io::Reader,
                          out: Box<io::Writer+'static>,
                          ann: &'a PpAnn,
                          is_expanded: bool) -> State<'a> {
        let (cmnts, lits) = comments::gather_comments_and_literals(
            span_diagnostic,
            filename,
            input);

        State::new(
            cm,
            out,
            ann,
            Some(cmnts),
            // If the code is post expansion, don't use the table of
            // literals, since it doesn't correspond with the literals
            // in the AST anymore.
            if is_expanded { None } else { Some(lits) })
    }

    pub fn new(cm: &'a CodeMap,
               out: Box<io::Writer+'static>,
               ann: &'a PpAnn,
               comments: Option<Vec<comments::Comment>>,
               literals: Option<Vec<comments::Literal>>) -> State<'a> {
        State {
            s: pp::mk_printer(out, default_columns),
            cm: Some(cm),
            comments: comments,
            literals: literals,
            cur_cmnt_and_lit: CurrentCommentAndLiteral {
                cur_cmnt: 0,
                cur_lit: 0
            },
            boxes: Vec::new(),
            ann: ann,
            encode_idents_with_hygiene: false,
        }
    }
}

pub fn to_string(f: |&mut State| -> IoResult<()>) -> String {
    use std::raw::TraitObject;
    let mut s = rust_printer(box MemWriter::new());
    f(&mut s).unwrap();
    eof(&mut s.s).unwrap();
    unsafe {
        // FIXME(pcwalton): A nasty function to extract the string from an `io::Writer`
        // that we "know" to be a `MemWriter` that works around the lack of checked
        // downcasts.
        let obj: TraitObject = mem::transmute_copy(&s.s.out);
        let wr: Box<MemWriter> = mem::transmute(obj.data);
        let result =
            String::from_utf8(Vec::from_slice(wr.get_ref().as_slice())).unwrap();
        mem::forget(wr);
        result.to_string()
    }
}

// FIXME (Issue #16472): the thing_to_string_impls macro should go away
// after we revise the syntax::ext::quote::ToToken impls to go directly
// to token-trees instead of thing -> string -> token-trees.

macro_rules! thing_to_string_impls {
    ($to_string:ident) => {

pub fn ty_to_string(ty: &ast::Ty) -> String {
    $to_string(|s| s.print_type(ty))
}

pub fn pat_to_string(pat: &ast::Pat) -> String {
    $to_string(|s| s.print_pat(pat))
}

pub fn arm_to_string(arm: &ast::Arm) -> String {
    $to_string(|s| s.print_arm(arm))
}

pub fn expr_to_string(e: &ast::Expr) -> String {
    $to_string(|s| s.print_expr(e))
}

pub fn lifetime_to_string(e: &ast::Lifetime) -> String {
    $to_string(|s| s.print_lifetime(e))
}

pub fn tt_to_string(tt: &ast::TokenTree) -> String {
    $to_string(|s| s.print_tt(tt))
}

pub fn tts_to_string(tts: &[ast::TokenTree]) -> String {
    $to_string(|s| s.print_tts(tts))
}

pub fn stmt_to_string(stmt: &ast::Stmt) -> String {
    $to_string(|s| s.print_stmt(stmt))
}

pub fn item_to_string(i: &ast::Item) -> String {
    $to_string(|s| s.print_item(i))
}

pub fn generics_to_string(generics: &ast::Generics) -> String {
    $to_string(|s| s.print_generics(generics))
}

pub fn ty_method_to_string(p: &ast::TypeMethod) -> String {
    $to_string(|s| s.print_ty_method(p))
}

pub fn method_to_string(p: &ast::Method) -> String {
    $to_string(|s| s.print_method(p))
}

pub fn fn_block_to_string(p: &ast::FnDecl) -> String {
    $to_string(|s| s.print_fn_block_args(p, None))
}

pub fn path_to_string(p: &ast::Path) -> String {
    $to_string(|s| s.print_path(p, false))
}

pub fn ident_to_string(id: &ast::Ident) -> String {
    $to_string(|s| s.print_ident(*id))
}

pub fn fun_to_string(decl: &ast::FnDecl, fn_style: ast::FnStyle, name: ast::Ident,
                  opt_explicit_self: Option<&ast::ExplicitSelf_>,
                  generics: &ast::Generics) -> String {
    $to_string(|s| {
        try!(s.print_fn(decl, Some(fn_style), abi::Rust,
                        name, generics, opt_explicit_self, ast::Inherited));
        try!(s.end()); // Close the head box
        s.end() // Close the outer box
    })
}

pub fn block_to_string(blk: &ast::Block) -> String {
    $to_string(|s| {
        // containing cbox, will be closed by print-block at }
        try!(s.cbox(indent_unit));
        // head-ibox, will be closed by print-block after {
        try!(s.ibox(0u));
        s.print_block(blk)
    })
}

pub fn meta_item_to_string(mi: &ast::MetaItem) -> String {
    $to_string(|s| s.print_meta_item(mi))
}

pub fn attribute_to_string(attr: &ast::Attribute) -> String {
    $to_string(|s| s.print_attribute(attr))
}

pub fn lit_to_string(l: &ast::Lit) -> String {
    $to_string(|s| s.print_literal(l))
}

pub fn explicit_self_to_string(explicit_self: &ast::ExplicitSelf_) -> String {
    $to_string(|s| s.print_explicit_self(explicit_self, ast::MutImmutable).map(|_| {}))
}

pub fn variant_to_string(var: &ast::Variant) -> String {
    $to_string(|s| s.print_variant(var))
}

pub fn arg_to_string(arg: &ast::Arg) -> String {
    $to_string(|s| s.print_arg(arg))
}

pub fn mac_to_string(arg: &ast::Mac) -> String {
    $to_string(|s| s.print_mac(arg))
}

} }

thing_to_string_impls!(to_string)

// FIXME (Issue #16472): the whole `with_hygiene` mod should go away
// after we revise the syntax::ext::quote::ToToken impls to go directly
// to token-trees instea of thing -> string -> token-trees.

pub mod with_hygiene {
    use abi;
    use ast;
    use std::io::IoResult;
    use super::indent_unit;

    // This function is the trick that all the rest of the routines
    // hang on.
    pub fn to_string_hyg(f: |&mut super::State| -> IoResult<()>) -> String {
        super::to_string(|s| {
            s.encode_idents_with_hygiene = true;
            f(s)
        })
    }

    thing_to_string_impls!(to_string_hyg)
}

pub fn visibility_qualified(vis: ast::Visibility, s: &str) -> String {
    match vis {
        ast::Public => format!("pub {}", s),
        ast::Inherited => s.to_string()
    }
}

fn needs_parentheses(expr: &ast::Expr) -> bool {
    match expr.node {
        ast::ExprAssign(..) | ast::ExprBinary(..) |
        ast::ExprFnBlock(..) | ast::ExprProc(..) |
        ast::ExprUnboxedFn(..) | ast::ExprAssignOp(..) |
        ast::ExprCast(..) => true,
        _ => false,
    }
}

impl<'a> State<'a> {
    pub fn ibox(&mut self, u: uint) -> IoResult<()> {
        self.boxes.push(pp::Inconsistent);
        pp::ibox(&mut self.s, u)
    }

    pub fn end(&mut self) -> IoResult<()> {
        self.boxes.pop().unwrap();
        pp::end(&mut self.s)
    }

    pub fn cbox(&mut self, u: uint) -> IoResult<()> {
        self.boxes.push(pp::Consistent);
        pp::cbox(&mut self.s, u)
    }

    // "raw box"
    pub fn rbox(&mut self, u: uint, b: pp::Breaks) -> IoResult<()> {
        self.boxes.push(b);
        pp::rbox(&mut self.s, u, b)
    }

    pub fn nbsp(&mut self) -> IoResult<()> { word(&mut self.s, " ") }

    pub fn word_nbsp(&mut self, w: &str) -> IoResult<()> {
        try!(word(&mut self.s, w));
        self.nbsp()
    }

    pub fn word_space(&mut self, w: &str) -> IoResult<()> {
        try!(word(&mut self.s, w));
        space(&mut self.s)
    }

    pub fn popen(&mut self) -> IoResult<()> { word(&mut self.s, "(") }

    pub fn pclose(&mut self) -> IoResult<()> { word(&mut self.s, ")") }

    pub fn head(&mut self, w: &str) -> IoResult<()> {
        // outer-box is consistent
        try!(self.cbox(indent_unit));
        // head-box is inconsistent
        try!(self.ibox(w.len() + 1));
        // keyword that starts the head
        if !w.is_empty() {
            try!(self.word_nbsp(w));
        }
        Ok(())
    }

    pub fn bopen(&mut self) -> IoResult<()> {
        try!(word(&mut self.s, "{"));
        self.end() // close the head-box
    }

    pub fn bclose_(&mut self, span: codemap::Span,
                   indented: uint) -> IoResult<()> {
        self.bclose_maybe_open(span, indented, true)
    }
    pub fn bclose_maybe_open (&mut self, span: codemap::Span,
                              indented: uint, close_box: bool) -> IoResult<()> {
        try!(self.maybe_print_comment(span.hi));
        try!(self.break_offset_if_not_bol(1u, -(indented as int)));
        try!(word(&mut self.s, "}"));
        if close_box {
            try!(self.end()); // close the outer-box
        }
        Ok(())
    }
    pub fn bclose(&mut self, span: codemap::Span) -> IoResult<()> {
        self.bclose_(span, indent_unit)
    }

    pub fn is_begin(&mut self) -> bool {
        match self.s.last_token() { pp::Begin(_) => true, _ => false }
    }

    pub fn is_end(&mut self) -> bool {
        match self.s.last_token() { pp::End => true, _ => false }
    }

    // is this the beginning of a line?
    pub fn is_bol(&mut self) -> bool {
        self.s.last_token().is_eof() || self.s.last_token().is_hardbreak_tok()
    }

    pub fn in_cbox(&self) -> bool {
        match self.boxes.last() {
            Some(&last_box) => last_box == pp::Consistent,
            None => false
        }
    }

    pub fn hardbreak_if_not_bol(&mut self) -> IoResult<()> {
        if !self.is_bol() {
            try!(hardbreak(&mut self.s))
        }
        Ok(())
    }
    pub fn space_if_not_bol(&mut self) -> IoResult<()> {
        if !self.is_bol() { try!(space(&mut self.s)); }
        Ok(())
    }
    pub fn break_offset_if_not_bol(&mut self, n: uint,
                                   off: int) -> IoResult<()> {
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
    pub fn synth_comment(&mut self, text: String) -> IoResult<()> {
        try!(word(&mut self.s, "/*"));
        try!(space(&mut self.s));
        try!(word(&mut self.s, text.as_slice()));
        try!(space(&mut self.s));
        word(&mut self.s, "*/")
    }

    pub fn commasep<T>(&mut self, b: Breaks, elts: &[T],
                       op: |&mut State, &T| -> IoResult<()>)
        -> IoResult<()> {
        try!(self.rbox(0u, b));
        let mut first = true;
        for elt in elts.iter() {
            if first { first = false; } else { try!(self.word_space(",")); }
            try!(op(self, elt));
        }
        self.end()
    }


    pub fn commasep_cmnt<T>(
                         &mut self,
                         b: Breaks,
                         elts: &[T],
                         op: |&mut State, &T| -> IoResult<()>,
                         get_span: |&T| -> codemap::Span) -> IoResult<()> {
        try!(self.rbox(0u, b));
        let len = elts.len();
        let mut i = 0u;
        for elt in elts.iter() {
            try!(self.maybe_print_comment(get_span(elt).hi));
            try!(op(self, elt));
            i += 1u;
            if i < len {
                try!(word(&mut self.s, ","));
                try!(self.maybe_print_trailing_comment(get_span(elt),
                                                    Some(get_span(&elts[i]).hi)));
                try!(self.space_if_not_bol());
            }
        }
        self.end()
    }

    pub fn commasep_exprs(&mut self, b: Breaks,
                          exprs: &[P<ast::Expr>]) -> IoResult<()> {
        self.commasep_cmnt(b, exprs, |s, e| s.print_expr(&**e), |e| e.span)
    }

    pub fn print_mod(&mut self, _mod: &ast::Mod,
                     attrs: &[ast::Attribute]) -> IoResult<()> {
        try!(self.print_inner_attributes(attrs));
        for vitem in _mod.view_items.iter() {
            try!(self.print_view_item(vitem));
        }
        for item in _mod.items.iter() {
            try!(self.print_item(&**item));
        }
        Ok(())
    }

    pub fn print_foreign_mod(&mut self, nmod: &ast::ForeignMod,
                             attrs: &[ast::Attribute]) -> IoResult<()> {
        try!(self.print_inner_attributes(attrs));
        for vitem in nmod.view_items.iter() {
            try!(self.print_view_item(vitem));
        }
        for item in nmod.items.iter() {
            try!(self.print_foreign_item(&**item));
        }
        Ok(())
    }

    pub fn print_opt_lifetime(&mut self,
                              lifetime: &Option<ast::Lifetime>) -> IoResult<()> {
        for l in lifetime.iter() {
            try!(self.print_lifetime(l));
            try!(self.nbsp());
        }
        Ok(())
    }

    pub fn print_type(&mut self, ty: &ast::Ty) -> IoResult<()> {
        try!(self.maybe_print_comment(ty.span.lo));
        try!(self.ibox(0u));
        match ty.node {
            ast::TyNil => try!(word(&mut self.s, "()")),
            ast::TyBot => try!(word(&mut self.s, "!")),
            ast::TyUniq(ref ty) => {
                try!(word(&mut self.s, "~"));
                try!(self.print_type(&**ty));
            }
            ast::TyVec(ref ty) => {
                try!(word(&mut self.s, "["));
                try!(self.print_type(&**ty));
                try!(word(&mut self.s, "]"));
            }
            ast::TyPtr(ref mt) => {
                try!(word(&mut self.s, "*"));
                match mt.mutbl {
                    ast::MutMutable => try!(self.word_nbsp("mut")),
                    ast::MutImmutable => try!(self.word_nbsp("const")),
                }
                try!(self.print_type(&*mt.ty));
            }
            ast::TyRptr(ref lifetime, ref mt) => {
                try!(word(&mut self.s, "&"));
                try!(self.print_opt_lifetime(lifetime));
                try!(self.print_mt(mt));
            }
            ast::TyTup(ref elts) => {
                try!(self.popen());
                try!(self.commasep(Inconsistent, elts.as_slice(),
                                   |s, ty| s.print_type(&**ty)));
                if elts.len() == 1 {
                    try!(word(&mut self.s, ","));
                }
                try!(self.pclose());
            }
            ast::TyParen(ref typ) => {
                try!(self.popen());
                try!(self.print_type(&**typ));
                try!(self.pclose());
            }
            ast::TyBareFn(ref f) => {
                let generics = ast::Generics {
                    lifetimes: f.lifetimes.clone(),
                    ty_params: OwnedSlice::empty(),
                    where_clause: ast::WhereClause {
                        id: ast::DUMMY_NODE_ID,
                        predicates: Vec::new(),
                    },
                };
                try!(self.print_ty_fn(Some(f.abi),
                                      None,
                                      f.fn_style,
                                      ast::Many,
                                      &*f.decl,
                                      None,
                                      &OwnedSlice::empty(),
                                      Some(&generics),
                                      None,
                                      None));
            }
            ast::TyClosure(ref f) => {
                let generics = ast::Generics {
                    lifetimes: f.lifetimes.clone(),
                    ty_params: OwnedSlice::empty(),
                    where_clause: ast::WhereClause {
                        id: ast::DUMMY_NODE_ID,
                        predicates: Vec::new(),
                    },
                };
                try!(self.print_ty_fn(None,
                                      Some('&'),
                                      f.fn_style,
                                      f.onceness,
                                      &*f.decl,
                                      None,
                                      &f.bounds,
                                      Some(&generics),
                                      None,
                                      None));
            }
            ast::TyProc(ref f) => {
                let generics = ast::Generics {
                    lifetimes: f.lifetimes.clone(),
                    ty_params: OwnedSlice::empty(),
                    where_clause: ast::WhereClause {
                        id: ast::DUMMY_NODE_ID,
                        predicates: Vec::new(),
                    },
                };
                try!(self.print_ty_fn(None,
                                      Some('~'),
                                      f.fn_style,
                                      f.onceness,
                                      &*f.decl,
                                      None,
                                      &f.bounds,
                                      Some(&generics),
                                      None,
                                      None));
            }
            ast::TyUnboxedFn(ref f) => {
                try!(self.print_ty_fn(None,
                                      None,
                                      ast::NormalFn,
                                      ast::Many,
                                      &*f.decl,
                                      None,
                                      &OwnedSlice::empty(),
                                      None,
                                      None,
                                      Some(f.kind)));
            }
            ast::TyPath(ref path, ref bounds, _) => {
                try!(self.print_bounded_path(path, bounds));
            }
            ast::TyQPath(ref qpath) => {
                try!(word(&mut self.s, "<"));
                try!(self.print_type(&*qpath.for_type));
                try!(space(&mut self.s));
                try!(self.word_space("as"));
                try!(self.print_path(&qpath.trait_name, false));
                try!(word(&mut self.s, ">"));
                try!(word(&mut self.s, "::"));
                try!(self.print_ident(qpath.item_name));
            }
            ast::TyFixedLengthVec(ref ty, ref v) => {
                try!(word(&mut self.s, "["));
                try!(self.print_type(&**ty));
                try!(word(&mut self.s, ", .."));
                try!(self.print_expr(&**v));
                try!(word(&mut self.s, "]"));
            }
            ast::TyTypeof(ref e) => {
                try!(word(&mut self.s, "typeof("));
                try!(self.print_expr(&**e));
                try!(word(&mut self.s, ")"));
            }
            ast::TyInfer => {
                try!(word(&mut self.s, "_"));
            }
        }
        self.end()
    }

    pub fn print_foreign_item(&mut self,
                              item: &ast::ForeignItem) -> IoResult<()> {
        try!(self.hardbreak_if_not_bol());
        try!(self.maybe_print_comment(item.span.lo));
        try!(self.print_outer_attributes(item.attrs.as_slice()));
        match item.node {
            ast::ForeignItemFn(ref decl, ref generics) => {
                try!(self.print_fn(&**decl, None, abi::Rust, item.ident, generics,
                                   None, item.vis));
                try!(self.end()); // end head-ibox
                try!(word(&mut self.s, ";"));
                self.end() // end the outer fn box
            }
            ast::ForeignItemStatic(ref t, m) => {
                try!(self.head(visibility_qualified(item.vis,
                                                    "static").as_slice()));
                if m {
                    try!(self.word_space("mut"));
                }
                try!(self.print_ident(item.ident));
                try!(self.word_space(":"));
                try!(self.print_type(&**t));
                try!(word(&mut self.s, ";"));
                try!(self.end()); // end the head-ibox
                self.end() // end the outer cbox
            }
        }
    }

    fn print_associated_type(&mut self, typedef: &ast::AssociatedType)
                             -> IoResult<()> {
        try!(self.word_space("type"));
        try!(self.print_ident(typedef.ident));
        word(&mut self.s, ";")
    }

    fn print_typedef(&mut self, typedef: &ast::Typedef) -> IoResult<()> {
        try!(self.word_space("type"));
        try!(self.print_ident(typedef.ident));
        try!(space(&mut self.s));
        try!(self.word_space("="));
        try!(self.print_type(&*typedef.typ));
        word(&mut self.s, ";")
    }

    /// Pretty-print an item
    pub fn print_item(&mut self, item: &ast::Item) -> IoResult<()> {
        try!(self.hardbreak_if_not_bol());
        try!(self.maybe_print_comment(item.span.lo));
        try!(self.print_outer_attributes(item.attrs.as_slice()));
        try!(self.ann.pre(self, NodeItem(item)));
        match item.node {
            ast::ItemStatic(ref ty, m, ref expr) => {
                try!(self.head(visibility_qualified(item.vis,
                                                    "static").as_slice()));
                if m == ast::MutMutable {
                    try!(self.word_space("mut"));
                }
                try!(self.print_ident(item.ident));
                try!(self.word_space(":"));
                try!(self.print_type(&**ty));
                try!(space(&mut self.s));
                try!(self.end()); // end the head-ibox

                try!(self.word_space("="));
                try!(self.print_expr(&**expr));
                try!(word(&mut self.s, ";"));
                try!(self.end()); // end the outer cbox
            }
            ast::ItemFn(ref decl, fn_style, abi, ref typarams, ref body) => {
                try!(self.print_fn(
                    &**decl,
                    Some(fn_style),
                    abi,
                    item.ident,
                    typarams,
                    None,
                    item.vis
                ));
                try!(word(&mut self.s, " "));
                try!(self.print_block_with_attrs(&**body, item.attrs.as_slice()));
            }
            ast::ItemMod(ref _mod) => {
                try!(self.head(visibility_qualified(item.vis,
                                                    "mod").as_slice()));
                try!(self.print_ident(item.ident));
                try!(self.nbsp());
                try!(self.bopen());
                try!(self.print_mod(_mod, item.attrs.as_slice()));
                try!(self.bclose(item.span));
            }
            ast::ItemForeignMod(ref nmod) => {
                try!(self.head("extern"));
                try!(self.word_nbsp(nmod.abi.to_string().as_slice()));
                try!(self.bopen());
                try!(self.print_foreign_mod(nmod, item.attrs.as_slice()));
                try!(self.bclose(item.span));
            }
            ast::ItemTy(ref ty, ref params) => {
                try!(self.ibox(indent_unit));
                try!(self.ibox(0u));
                try!(self.word_nbsp(visibility_qualified(item.vis,
                                                         "type").as_slice()));
                try!(self.print_ident(item.ident));
                try!(self.print_generics(params));
                try!(self.end()); // end the inner ibox

                try!(space(&mut self.s));
                try!(self.word_space("="));
                try!(self.print_type(&**ty));
                try!(self.print_where_clause(params));
                try!(word(&mut self.s, ";"));
                try!(self.end()); // end the outer ibox
            }
            ast::ItemEnum(ref enum_definition, ref params) => {
                try!(self.print_enum_def(
                    enum_definition,
                    params,
                    item.ident,
                    item.span,
                    item.vis
                ));
            }
            ast::ItemStruct(ref struct_def, ref generics) => {
                if struct_def.is_virtual {
                    try!(self.word_space("virtual"));
                }
                try!(self.head(visibility_qualified(item.vis,"struct").as_slice()));
                try!(self.print_struct(&**struct_def, generics, item.ident, item.span));
            }

            ast::ItemImpl(ref generics,
                          ref opt_trait,
                          ref ty,
                          ref impl_items) => {
                try!(self.head(visibility_qualified(item.vis,
                                                    "impl").as_slice()));
                if generics.is_parameterized() {
                    try!(self.print_generics(generics));
                    try!(space(&mut self.s));
                }

                match opt_trait {
                    &Some(ref t) => {
                        try!(self.print_trait_ref(t));
                        try!(space(&mut self.s));
                        try!(self.word_space("for"));
                    }
                    &None => {}
                }

                try!(self.print_type(&**ty));
                try!(self.print_where_clause(generics));

                try!(space(&mut self.s));
                try!(self.bopen());
                try!(self.print_inner_attributes(item.attrs.as_slice()));
                for impl_item in impl_items.iter() {
                    match *impl_item {
                        ast::MethodImplItem(ref meth) => {
                            try!(self.print_method(&**meth));
                        }
                        ast::TypeImplItem(ref typ) => {
                            try!(self.print_typedef(&**typ));
                        }
                    }
                }
                try!(self.bclose(item.span));
            }
            ast::ItemTrait(ref generics, ref unbound, ref bounds, ref methods) => {
                try!(self.head(visibility_qualified(item.vis,
                                                    "trait").as_slice()));
                try!(self.print_ident(item.ident));
                try!(self.print_generics(generics));
                match unbound {
                    &Some(TraitTyParamBound(ref tref)) => {
                        try!(space(&mut self.s));
                        try!(self.word_space("for"));
                        try!(self.print_trait_ref(tref));
                        try!(word(&mut self.s, "?"));
                    }
                    _ => {}
                }
                try!(self.print_bounds(":", bounds));
                try!(self.print_where_clause(generics));
                try!(word(&mut self.s, " "));
                try!(self.bopen());
                for meth in methods.iter() {
                    try!(self.print_trait_method(meth));
                }
                try!(self.bclose(item.span));
            }
            // I think it's reasonable to hide the context here:
            ast::ItemMac(codemap::Spanned { node: ast::MacInvocTT(ref pth, ref tts, _),
                                            ..}) => {
                try!(self.print_visibility(item.vis));
                try!(self.print_path(pth, false));
                try!(word(&mut self.s, "! "));
                try!(self.print_ident(item.ident));
                try!(self.cbox(indent_unit));
                try!(self.popen());
                try!(self.print_tts(tts.as_slice()));
                try!(self.pclose());
                try!(self.end());
            }
        }
        self.ann.post(self, NodeItem(item))
    }

    fn print_trait_ref(&mut self, t: &ast::TraitRef) -> IoResult<()> {
        if t.lifetimes.len() > 0 {
            try!(self.print_generics(&ast::Generics {
                lifetimes: t.lifetimes.clone(),
                ty_params: OwnedSlice::empty(),
                where_clause: ast::WhereClause {
                    id: ast::DUMMY_NODE_ID,
                    predicates: Vec::new(),
                },
            }));
        }
        self.print_path(&t.path, false)
    }

    pub fn print_enum_def(&mut self, enum_definition: &ast::EnumDef,
                          generics: &ast::Generics, ident: ast::Ident,
                          span: codemap::Span,
                          visibility: ast::Visibility) -> IoResult<()> {
        try!(self.head(visibility_qualified(visibility, "enum").as_slice()));
        try!(self.print_ident(ident));
        try!(self.print_generics(generics));
        try!(self.print_where_clause(generics));
        try!(space(&mut self.s));
        self.print_variants(enum_definition.variants.as_slice(), span)
    }

    pub fn print_variants(&mut self,
                          variants: &[P<ast::Variant>],
                          span: codemap::Span) -> IoResult<()> {
        try!(self.bopen());
        for v in variants.iter() {
            try!(self.space_if_not_bol());
            try!(self.maybe_print_comment(v.span.lo));
            try!(self.print_outer_attributes(v.node.attrs.as_slice()));
            try!(self.ibox(indent_unit));
            try!(self.print_variant(&**v));
            try!(word(&mut self.s, ","));
            try!(self.end());
            try!(self.maybe_print_trailing_comment(v.span, None));
        }
        self.bclose(span)
    }

    pub fn print_visibility(&mut self, vis: ast::Visibility) -> IoResult<()> {
        match vis {
            ast::Public => self.word_nbsp("pub"),
            ast::Inherited => Ok(())
        }
    }

    pub fn print_struct(&mut self,
                        struct_def: &ast::StructDef,
                        generics: &ast::Generics,
                        ident: ast::Ident,
                        span: codemap::Span) -> IoResult<()> {
        try!(self.print_ident(ident));
        try!(self.print_generics(generics));
        match struct_def.super_struct {
            Some(ref t) => {
                try!(self.word_space(":"));
                try!(self.print_type(&**t));
            },
            None => {},
        }
        if ast_util::struct_def_is_tuple_like(struct_def) {
            if !struct_def.fields.is_empty() {
                try!(self.popen());
                try!(self.commasep(
                    Inconsistent, struct_def.fields.as_slice(),
                    |s, field| {
                        match field.node.kind {
                            ast::NamedField(..) => fail!("unexpected named field"),
                            ast::UnnamedField(vis) => {
                                try!(s.print_visibility(vis));
                                try!(s.maybe_print_comment(field.span.lo));
                                s.print_type(&*field.node.ty)
                            }
                        }
                    }
                ));
                try!(self.pclose());
            }
            try!(word(&mut self.s, ";"));
            try!(self.end());
            self.end() // close the outer-box
        } else {
            try!(self.nbsp());
            try!(self.bopen());
            try!(self.hardbreak_if_not_bol());

            for field in struct_def.fields.iter() {
                match field.node.kind {
                    ast::UnnamedField(..) => fail!("unexpected unnamed field"),
                    ast::NamedField(ident, visibility) => {
                        try!(self.hardbreak_if_not_bol());
                        try!(self.maybe_print_comment(field.span.lo));
                        try!(self.print_outer_attributes(field.node.attrs.as_slice()));
                        try!(self.print_visibility(visibility));
                        try!(self.print_ident(ident));
                        try!(self.word_nbsp(":"));
                        try!(self.print_type(&*field.node.ty));
                        try!(word(&mut self.s, ","));
                    }
                }
            }

            self.bclose(span)
        }
    }

    /// This doesn't deserve to be called "pretty" printing, but it should be
    /// meaning-preserving. A quick hack that might help would be to look at the
    /// spans embedded in the TTs to decide where to put spaces and newlines.
    /// But it'd be better to parse these according to the grammar of the
    /// appropriate macro, transcribe back into the grammar we just parsed from,
    /// and then pretty-print the resulting AST nodes (so, e.g., we print
    /// expression arguments as expressions). It can be done! I think.
    pub fn print_tt(&mut self, tt: &ast::TokenTree) -> IoResult<()> {
        match *tt {
            ast::TTDelim(ref tts) => self.print_tts(tts.as_slice()),
            ast::TTTok(_, ref tk) => {
                try!(word(&mut self.s, parse::token::to_string(tk).as_slice()));
                match *tk {
                    parse::token::DOC_COMMENT(..) => {
                        hardbreak(&mut self.s)
                    }
                    _ => Ok(())
                }
            }
            ast::TTSeq(_, ref tts, ref sep, zerok) => {
                try!(word(&mut self.s, "$("));
                for tt_elt in (*tts).iter() {
                    try!(self.print_tt(tt_elt));
                }
                try!(word(&mut self.s, ")"));
                match *sep {
                    Some(ref tk) => {
                        try!(word(&mut self.s,
                                  parse::token::to_string(tk).as_slice()));
                    }
                    None => ()
                }
                word(&mut self.s, if zerok { "*" } else { "+" })
            }
            ast::TTNonterminal(_, name) => {
                try!(word(&mut self.s, "$"));
                self.print_ident(name)
            }
        }
    }

    pub fn print_tts(&mut self, tts: &[ast::TokenTree]) -> IoResult<()> {
        try!(self.ibox(0));
        for (i, tt) in tts.iter().enumerate() {
            if i != 0 {
                try!(space(&mut self.s));
            }
            try!(self.print_tt(tt));
        }
        self.end()
    }

    pub fn print_variant(&mut self, v: &ast::Variant) -> IoResult<()> {
        try!(self.print_visibility(v.node.vis));
        match v.node.kind {
            ast::TupleVariantKind(ref args) => {
                try!(self.print_ident(v.node.name));
                if !args.is_empty() {
                    try!(self.popen());
                    try!(self.commasep(Consistent,
                                       args.as_slice(),
                                       |s, arg| s.print_type(&*arg.ty)));
                    try!(self.pclose());
                }
            }
            ast::StructVariantKind(ref struct_def) => {
                try!(self.head(""));
                let generics = ast_util::empty_generics();
                try!(self.print_struct(&**struct_def, &generics, v.node.name, v.span));
            }
        }
        match v.node.disr_expr {
            Some(ref d) => {
                try!(space(&mut self.s));
                try!(self.word_space("="));
                self.print_expr(&**d)
            }
            _ => Ok(())
        }
    }

    pub fn print_ty_method(&mut self, m: &ast::TypeMethod) -> IoResult<()> {
        try!(self.hardbreak_if_not_bol());
        try!(self.maybe_print_comment(m.span.lo));
        try!(self.print_outer_attributes(m.attrs.as_slice()));
        try!(self.print_ty_fn(None,
                              None,
                              m.fn_style,
                              ast::Many,
                              &*m.decl,
                              Some(m.ident),
                              &OwnedSlice::empty(),
                              Some(&m.generics),
                              Some(&m.explicit_self.node),
                              None));
        word(&mut self.s, ";")
    }

    pub fn print_trait_method(&mut self,
                              m: &ast::TraitItem) -> IoResult<()> {
        match *m {
            RequiredMethod(ref ty_m) => self.print_ty_method(ty_m),
            ProvidedMethod(ref m) => self.print_method(&**m),
            TypeTraitItem(ref t) => self.print_associated_type(&**t),
        }
    }

    pub fn print_impl_item(&mut self, ii: &ast::ImplItem) -> IoResult<()> {
        match *ii {
            MethodImplItem(ref m) => self.print_method(&**m),
            TypeImplItem(ref td) => self.print_typedef(&**td),
        }
    }

    pub fn print_method(&mut self, meth: &ast::Method) -> IoResult<()> {
        try!(self.hardbreak_if_not_bol());
        try!(self.maybe_print_comment(meth.span.lo));
        try!(self.print_outer_attributes(meth.attrs.as_slice()));
        match meth.node {
            ast::MethDecl(ident,
                          ref generics,
                          abi,
                          ref explicit_self,
                          fn_style,
                          ref decl,
                          ref body,
                          vis) => {
                try!(self.print_fn(&**decl,
                                   Some(fn_style),
                                   abi,
                                   ident,
                                   generics,
                                   Some(&explicit_self.node),
                                   vis));
                try!(word(&mut self.s, " "));
                self.print_block_with_attrs(&**body, meth.attrs.as_slice())
            },
            ast::MethMac(codemap::Spanned { node: ast::MacInvocTT(ref pth, ref tts, _),
                                            ..}) => {
                // code copied from ItemMac:
                try!(self.print_path(pth, false));
                try!(word(&mut self.s, "! "));
                try!(self.cbox(indent_unit));
                try!(self.popen());
                try!(self.print_tts(tts.as_slice()));
                try!(self.pclose());
                self.end()
            }
        }
    }

    pub fn print_outer_attributes(&mut self,
                                  attrs: &[ast::Attribute]) -> IoResult<()> {
        let mut count = 0u;
        for attr in attrs.iter() {
            match attr.node.style {
                ast::AttrOuter => {
                    try!(self.print_attribute(attr));
                    count += 1;
                }
                _ => {/* fallthrough */ }
            }
        }
        if count > 0 {
            try!(self.hardbreak_if_not_bol());
        }
        Ok(())
    }

    pub fn print_inner_attributes(&mut self,
                                  attrs: &[ast::Attribute]) -> IoResult<()> {
        let mut count = 0u;
        for attr in attrs.iter() {
            match attr.node.style {
                ast::AttrInner => {
                    try!(self.print_attribute(attr));
                    count += 1;
                }
                _ => {/* fallthrough */ }
            }
        }
        if count > 0 {
            try!(self.hardbreak_if_not_bol());
        }
        Ok(())
    }

    pub fn print_attribute(&mut self, attr: &ast::Attribute) -> IoResult<()> {
        try!(self.hardbreak_if_not_bol());
        try!(self.maybe_print_comment(attr.span.lo));
        if attr.node.is_sugared_doc {
            word(&mut self.s, attr.value_str().unwrap().get())
        } else {
            match attr.node.style {
                ast::AttrInner => try!(word(&mut self.s, "#![")),
                ast::AttrOuter => try!(word(&mut self.s, "#[")),
            }
            try!(self.print_meta_item(&*attr.meta()));
            word(&mut self.s, "]")
        }
    }


    pub fn print_stmt(&mut self, st: &ast::Stmt) -> IoResult<()> {
        try!(self.maybe_print_comment(st.span.lo));
        match st.node {
            ast::StmtDecl(ref decl, _) => {
                try!(self.print_decl(&**decl));
            }
            ast::StmtExpr(ref expr, _) => {
                try!(self.space_if_not_bol());
                try!(self.print_expr(&**expr));
            }
            ast::StmtSemi(ref expr, _) => {
                try!(self.space_if_not_bol());
                try!(self.print_expr(&**expr));
                try!(word(&mut self.s, ";"));
            }
            ast::StmtMac(ref mac, semi) => {
                try!(self.space_if_not_bol());
                try!(self.print_mac(mac));
                if semi {
                    try!(word(&mut self.s, ";"));
                }
            }
        }
        if parse::classify::stmt_ends_with_semi(&st.node) {
            try!(word(&mut self.s, ";"));
        }
        self.maybe_print_trailing_comment(st.span, None)
    }

    pub fn print_block(&mut self, blk: &ast::Block) -> IoResult<()> {
        self.print_block_with_attrs(blk, &[])
    }

    pub fn print_block_unclosed(&mut self, blk: &ast::Block) -> IoResult<()> {
        self.print_block_unclosed_indent(blk, indent_unit)
    }

    pub fn print_block_unclosed_indent(&mut self, blk: &ast::Block,
                                       indented: uint) -> IoResult<()> {
        self.print_block_maybe_unclosed(blk, indented, &[], false)
    }

    pub fn print_block_with_attrs(&mut self,
                                  blk: &ast::Block,
                                  attrs: &[ast::Attribute]) -> IoResult<()> {
        self.print_block_maybe_unclosed(blk, indent_unit, attrs, true)
    }

    pub fn print_block_maybe_unclosed(&mut self,
                                      blk: &ast::Block,
                                      indented: uint,
                                      attrs: &[ast::Attribute],
                                      close_box: bool) -> IoResult<()> {
        match blk.rules {
            ast::UnsafeBlock(..) => try!(self.word_space("unsafe")),
            ast::DefaultBlock => ()
        }
        try!(self.maybe_print_comment(blk.span.lo));
        try!(self.ann.pre(self, NodeBlock(blk)));
        try!(self.bopen());

        try!(self.print_inner_attributes(attrs));

        for vi in blk.view_items.iter() {
            try!(self.print_view_item(vi));
        }
        for st in blk.stmts.iter() {
            try!(self.print_stmt(&**st));
        }
        match blk.expr {
            Some(ref expr) => {
                try!(self.space_if_not_bol());
                try!(self.print_expr(&**expr));
                try!(self.maybe_print_trailing_comment(expr.span, Some(blk.span.hi)));
            }
            _ => ()
        }
        try!(self.bclose_maybe_open(blk.span, indented, close_box));
        self.ann.post(self, NodeBlock(blk))
    }

    fn print_else(&mut self, els: Option<&ast::Expr>) -> IoResult<()> {
        match els {
            Some(_else) => {
                match _else.node {
                    // "another else-if"
                    ast::ExprIf(ref i, ref then, ref e) => {
                        try!(self.cbox(indent_unit - 1u));
                        try!(self.ibox(0u));
                        try!(word(&mut self.s, " else if "));
                        try!(self.print_expr(&**i));
                        try!(space(&mut self.s));
                        try!(self.print_block(&**then));
                        self.print_else(e.as_ref().map(|e| &**e))
                    }
                    // "another else-if-let"
                    ast::ExprIfLet(ref pat, ref expr, ref then, ref e) => {
                        try!(self.cbox(indent_unit - 1u));
                        try!(self.ibox(0u));
                        try!(word(&mut self.s, " else if let "));
                        try!(self.print_pat(&**pat));
                        try!(space(&mut self.s));
                        try!(self.word_space("="));
                        try!(self.print_expr(&**expr));
                        try!(space(&mut self.s));
                        try!(self.print_block(&**then));
                        self.print_else(e.as_ref().map(|e| &**e))
                    }
                    // "final else"
                    ast::ExprBlock(ref b) => {
                        try!(self.cbox(indent_unit - 1u));
                        try!(self.ibox(0u));
                        try!(word(&mut self.s, " else "));
                        self.print_block(&**b)
                    }
                    // BLEAH, constraints would be great here
                    _ => {
                        fail!("print_if saw if with weird alternative");
                    }
                }
            }
            _ => Ok(())
        }
    }

    pub fn print_if(&mut self, test: &ast::Expr, blk: &ast::Block,
                    elseopt: Option<&ast::Expr>) -> IoResult<()> {
        try!(self.head("if"));
        try!(self.print_expr(test));
        try!(space(&mut self.s));
        try!(self.print_block(blk));
        self.print_else(elseopt)
    }

    pub fn print_if_let(&mut self, pat: &ast::Pat, expr: &ast::Expr, blk: &ast::Block,
                        elseopt: Option<&ast::Expr>) -> IoResult<()> {
        try!(self.head("if let"));
        try!(self.print_pat(pat));
        try!(space(&mut self.s));
        try!(self.word_space("="));
        try!(self.print_expr(expr));
        try!(space(&mut self.s));
        try!(self.print_block(blk));
        self.print_else(elseopt)
    }

    pub fn print_mac(&mut self, m: &ast::Mac) -> IoResult<()> {
        match m.node {
            // I think it's reasonable to hide the ctxt here:
            ast::MacInvocTT(ref pth, ref tts, _) => {
                try!(self.print_path(pth, false));
                try!(word(&mut self.s, "!"));
                try!(self.popen());
                try!(self.print_tts(tts.as_slice()));
                self.pclose()
            }
        }
    }


    fn print_call_post(&mut self, args: &[P<ast::Expr>]) -> IoResult<()> {
        try!(self.popen());
        try!(self.commasep_exprs(Inconsistent, args));
        self.pclose()
    }

    pub fn print_expr_maybe_paren(&mut self, expr: &ast::Expr) -> IoResult<()> {
        let needs_par = needs_parentheses(expr);
        if needs_par {
            try!(self.popen());
        }
        try!(self.print_expr(expr));
        if needs_par {
            try!(self.pclose());
        }
        Ok(())
    }

    pub fn print_expr(&mut self, expr: &ast::Expr) -> IoResult<()> {
        try!(self.maybe_print_comment(expr.span.lo));
        try!(self.ibox(indent_unit));
        try!(self.ann.pre(self, NodeExpr(expr)));
        match expr.node {
            ast::ExprBox(ref p, ref e) => {
                try!(word(&mut self.s, "box"));
                try!(word(&mut self.s, "("));
                try!(self.print_expr(&**p));
                try!(self.word_space(")"));
                try!(self.print_expr(&**e));
            }
            ast::ExprVec(ref exprs) => {
                try!(self.ibox(indent_unit));
                try!(word(&mut self.s, "["));
                try!(self.commasep_exprs(Inconsistent, exprs.as_slice()));
                try!(word(&mut self.s, "]"));
                try!(self.end());
            }

            ast::ExprRepeat(ref element, ref count) => {
                try!(self.ibox(indent_unit));
                try!(word(&mut self.s, "["));
                try!(self.print_expr(&**element));
                try!(word(&mut self.s, ","));
                try!(word(&mut self.s, ".."));
                try!(self.print_expr(&**count));
                try!(word(&mut self.s, "]"));
                try!(self.end());
            }

            ast::ExprStruct(ref path, ref fields, ref wth) => {
                try!(self.print_path(path, true));
                try!(word(&mut self.s, "{"));
                try!(self.commasep_cmnt(
                    Consistent,
                    fields.as_slice(),
                    |s, field| {
                        try!(s.ibox(indent_unit));
                        try!(s.print_ident(field.ident.node));
                        try!(s.word_space(":"));
                        try!(s.print_expr(&*field.expr));
                        s.end()
                    },
                    |f| f.span));
                match *wth {
                    Some(ref expr) => {
                        try!(self.ibox(indent_unit));
                        if !fields.is_empty() {
                            try!(word(&mut self.s, ","));
                            try!(space(&mut self.s));
                        }
                        try!(word(&mut self.s, ".."));
                        try!(self.print_expr(&**expr));
                        try!(self.end());
                    }
                    _ => try!(word(&mut self.s, ","))
                }
                try!(word(&mut self.s, "}"));
            }
            ast::ExprTup(ref exprs) => {
                try!(self.popen());
                try!(self.commasep_exprs(Inconsistent, exprs.as_slice()));
                if exprs.len() == 1 {
                    try!(word(&mut self.s, ","));
                }
                try!(self.pclose());
            }
            ast::ExprCall(ref func, ref args) => {
                try!(self.print_expr_maybe_paren(&**func));
                try!(self.print_call_post(args.as_slice()));
            }
            ast::ExprMethodCall(ident, ref tys, ref args) => {
                let base_args = args.slice_from(1);
                try!(self.print_expr(&**args.get(0)));
                try!(word(&mut self.s, "."));
                try!(self.print_ident(ident.node));
                if tys.len() > 0u {
                    try!(word(&mut self.s, "::<"));
                    try!(self.commasep(Inconsistent, tys.as_slice(),
                                       |s, ty| s.print_type(&**ty)));
                    try!(word(&mut self.s, ">"));
                }
                try!(self.print_call_post(base_args));
            }
            ast::ExprBinary(op, ref lhs, ref rhs) => {
                try!(self.print_expr(&**lhs));
                try!(space(&mut self.s));
                try!(self.word_space(ast_util::binop_to_string(op)));
                try!(self.print_expr(&**rhs));
            }
            ast::ExprUnary(op, ref expr) => {
                try!(word(&mut self.s, ast_util::unop_to_string(op)));
                try!(self.print_expr_maybe_paren(&**expr));
            }
            ast::ExprAddrOf(m, ref expr) => {
                try!(word(&mut self.s, "&"));
                try!(self.print_mutability(m));
                try!(self.print_expr_maybe_paren(&**expr));
            }
            ast::ExprLit(ref lit) => try!(self.print_literal(&**lit)),
            ast::ExprCast(ref expr, ref ty) => {
                try!(self.print_expr(&**expr));
                try!(space(&mut self.s));
                try!(self.word_space("as"));
                try!(self.print_type(&**ty));
            }
            ast::ExprIf(ref test, ref blk, ref elseopt) => {
                try!(self.print_if(&**test, &**blk, elseopt.as_ref().map(|e| &**e)));
            }
            ast::ExprIfLet(ref pat, ref expr, ref blk, ref elseopt) => {
                try!(self.print_if_let(&**pat, &**expr, &** blk, elseopt.as_ref().map(|e| &**e)));
            }
            ast::ExprWhile(ref test, ref blk, opt_ident) => {
                for ident in opt_ident.iter() {
                    try!(self.print_ident(*ident));
                    try!(self.word_space(":"));
                }
                try!(self.head("while"));
                try!(self.print_expr(&**test));
                try!(space(&mut self.s));
                try!(self.print_block(&**blk));
            }
            ast::ExprForLoop(ref pat, ref iter, ref blk, opt_ident) => {
                for ident in opt_ident.iter() {
                    try!(self.print_ident(*ident));
                    try!(self.word_space(":"));
                }
                try!(self.head("for"));
                try!(self.print_pat(&**pat));
                try!(space(&mut self.s));
                try!(self.word_space("in"));
                try!(self.print_expr(&**iter));
                try!(space(&mut self.s));
                try!(self.print_block(&**blk));
            }
            ast::ExprLoop(ref blk, opt_ident) => {
                for ident in opt_ident.iter() {
                    try!(self.print_ident(*ident));
                    try!(self.word_space(":"));
                }
                try!(self.head("loop"));
                try!(space(&mut self.s));
                try!(self.print_block(&**blk));
            }
            ast::ExprMatch(ref expr, ref arms, _) => {
                try!(self.cbox(indent_unit));
                try!(self.ibox(4));
                try!(self.word_nbsp("match"));
                try!(self.print_expr(&**expr));
                try!(space(&mut self.s));
                try!(self.bopen());
                for arm in arms.iter() {
                    try!(self.print_arm(arm));
                }
                try!(self.bclose_(expr.span, indent_unit));
            }
            ast::ExprFnBlock(capture_clause, ref decl, ref body) => {
                try!(self.print_capture_clause(capture_clause));

                // in do/for blocks we don't want to show an empty
                // argument list, but at this point we don't know which
                // we are inside.
                //
                // if !decl.inputs.is_empty() {
                try!(self.print_fn_block_args(&**decl, None));
                try!(space(&mut self.s));
                // }

                if !body.stmts.is_empty() || !body.expr.is_some() {
                    try!(self.print_block_unclosed(&**body));
                } else {
                    // we extract the block, so as not to create another set of boxes
                    match body.expr.as_ref().unwrap().node {
                        ast::ExprBlock(ref blk) => {
                            try!(self.print_block_unclosed(&**blk));
                        }
                        _ => {
                            // this is a bare expression
                            try!(self.print_expr(&**body.expr.as_ref().unwrap()));
                            try!(self.end()); // need to close a box
                        }
                    }
                }
                // a box will be closed by print_expr, but we didn't want an overall
                // wrapper so we closed the corresponding opening. so create an
                // empty box to satisfy the close.
                try!(self.ibox(0));
            }
            ast::ExprUnboxedFn(capture_clause, kind, ref decl, ref body) => {
                try!(self.print_capture_clause(capture_clause));

                // in do/for blocks we don't want to show an empty
                // argument list, but at this point we don't know which
                // we are inside.
                //
                // if !decl.inputs.is_empty() {
                try!(self.print_fn_block_args(&**decl, Some(kind)));
                try!(space(&mut self.s));
                // }

                if !body.stmts.is_empty() || !body.expr.is_some() {
                    try!(self.print_block_unclosed(&**body));
                } else {
                    // we extract the block, so as not to create another set of boxes
                    match body.expr.as_ref().unwrap().node {
                        ast::ExprBlock(ref blk) => {
                            try!(self.print_block_unclosed(&**blk));
                        }
                        _ => {
                            // this is a bare expression
                            try!(self.print_expr(body.expr.as_ref().map(|e| &**e).unwrap()));
                            try!(self.end()); // need to close a box
                        }
                    }
                }
                // a box will be closed by print_expr, but we didn't want an overall
                // wrapper so we closed the corresponding opening. so create an
                // empty box to satisfy the close.
                try!(self.ibox(0));
            }
            ast::ExprProc(ref decl, ref body) => {
                // in do/for blocks we don't want to show an empty
                // argument list, but at this point we don't know which
                // we are inside.
                //
                // if !decl.inputs.is_empty() {
                try!(self.print_proc_args(&**decl));
                try!(space(&mut self.s));
                // }
                assert!(body.stmts.is_empty());
                assert!(body.expr.is_some());
                // we extract the block, so as not to create another set of boxes
                match body.expr.as_ref().unwrap().node {
                    ast::ExprBlock(ref blk) => {
                        try!(self.print_block_unclosed(&**blk));
                    }
                    _ => {
                        // this is a bare expression
                        try!(self.print_expr(body.expr.as_ref().map(|e| &**e).unwrap()));
                        try!(self.end()); // need to close a box
                    }
                }
                // a box will be closed by print_expr, but we didn't want an overall
                // wrapper so we closed the corresponding opening. so create an
                // empty box to satisfy the close.
                try!(self.ibox(0));
            }
            ast::ExprBlock(ref blk) => {
                // containing cbox, will be closed by print-block at }
                try!(self.cbox(indent_unit));
                // head-box, will be closed by print-block after {
                try!(self.ibox(0u));
                try!(self.print_block(&**blk));
            }
            ast::ExprAssign(ref lhs, ref rhs) => {
                try!(self.print_expr(&**lhs));
                try!(space(&mut self.s));
                try!(self.word_space("="));
                try!(self.print_expr(&**rhs));
            }
            ast::ExprAssignOp(op, ref lhs, ref rhs) => {
                try!(self.print_expr(&**lhs));
                try!(space(&mut self.s));
                try!(word(&mut self.s, ast_util::binop_to_string(op)));
                try!(self.word_space("="));
                try!(self.print_expr(&**rhs));
            }
            ast::ExprField(ref expr, id, ref tys) => {
                try!(self.print_expr(&**expr));
                try!(word(&mut self.s, "."));
                try!(self.print_ident(id.node));
                if tys.len() > 0u {
                    try!(word(&mut self.s, "::<"));
                    try!(self.commasep(
                        Inconsistent, tys.as_slice(),
                        |s, ty| s.print_type(&**ty)));
                    try!(word(&mut self.s, ">"));
                }
            }
            ast::ExprTupField(ref expr, id, ref tys) => {
                try!(self.print_expr(&**expr));
                try!(word(&mut self.s, "."));
                try!(self.print_uint(id.node));
                if tys.len() > 0u {
                    try!(word(&mut self.s, "::<"));
                    try!(self.commasep(
                        Inconsistent, tys.as_slice(),
                        |s, ty| s.print_type(&**ty)));
                    try!(word(&mut self.s, ">"));
                }
            }
            ast::ExprIndex(ref expr, ref index) => {
                try!(self.print_expr(&**expr));
                try!(word(&mut self.s, "["));
                try!(self.print_expr(&**index));
                try!(word(&mut self.s, "]"));
            }
            ast::ExprSlice(ref e, ref start, ref end, ref mutbl) => {
                try!(self.print_expr(&**e));
                try!(word(&mut self.s, "["));
                if mutbl == &ast::MutMutable {
                    try!(word(&mut self.s, "mut"));
                    if start.is_some() || end.is_some() {
                        try!(space(&mut self.s));
                    }
                }
                match start {
                    &Some(ref e) => try!(self.print_expr(&**e)),
                    _ => {}
                }
                if start.is_some() || end.is_some() {
                    try!(word(&mut self.s, ".."));
                }
                match end {
                    &Some(ref e) => try!(self.print_expr(&**e)),
                    _ => {}
                }
                try!(word(&mut self.s, "]"));
            }
            ast::ExprPath(ref path) => try!(self.print_path(path, true)),
            ast::ExprBreak(opt_ident) => {
                try!(word(&mut self.s, "break"));
                try!(space(&mut self.s));
                for ident in opt_ident.iter() {
                    try!(self.print_ident(*ident));
                    try!(space(&mut self.s));
                }
            }
            ast::ExprAgain(opt_ident) => {
                try!(word(&mut self.s, "continue"));
                try!(space(&mut self.s));
                for ident in opt_ident.iter() {
                    try!(self.print_ident(*ident));
                    try!(space(&mut self.s))
                }
            }
            ast::ExprRet(ref result) => {
                try!(word(&mut self.s, "return"));
                match *result {
                    Some(ref expr) => {
                        try!(word(&mut self.s, " "));
                        try!(self.print_expr(&**expr));
                    }
                    _ => ()
                }
            }
            ast::ExprInlineAsm(ref a) => {
                if a.volatile {
                    try!(word(&mut self.s, "__volatile__ asm!"));
                } else {
                    try!(word(&mut self.s, "asm!"));
                }
                try!(self.popen());
                try!(self.print_string(a.asm.get(), a.asm_str_style));
                try!(self.word_space(":"));

                try!(self.commasep(Inconsistent, a.outputs.as_slice(),
                                   |s, &(ref co, ref o, is_rw)| {
                    match co.get().slice_shift_char() {
                        (Some('='), operand) if is_rw => {
                            try!(s.print_string(format!("+{}", operand).as_slice(),
                                                ast::CookedStr))
                        }
                        _ => try!(s.print_string(co.get(), ast::CookedStr))
                    }
                    try!(s.popen());
                    try!(s.print_expr(&**o));
                    try!(s.pclose());
                    Ok(())
                }));
                try!(space(&mut self.s));
                try!(self.word_space(":"));

                try!(self.commasep(Inconsistent, a.inputs.as_slice(),
                                   |s, &(ref co, ref o)| {
                    try!(s.print_string(co.get(), ast::CookedStr));
                    try!(s.popen());
                    try!(s.print_expr(&**o));
                    try!(s.pclose());
                    Ok(())
                }));
                try!(space(&mut self.s));
                try!(self.word_space(":"));

                try!(self.print_string(a.clobbers.get(), ast::CookedStr));
                try!(self.pclose());
            }
            ast::ExprMac(ref m) => try!(self.print_mac(m)),
            ast::ExprParen(ref e) => {
                try!(self.popen());
                try!(self.print_expr(&**e));
                try!(self.pclose());
            }
        }
        try!(self.ann.post(self, NodeExpr(expr)));
        self.end()
    }

    pub fn print_local_decl(&mut self, loc: &ast::Local) -> IoResult<()> {
        try!(self.print_pat(&*loc.pat));
        match loc.ty.node {
            ast::TyInfer => Ok(()),
            _ => {
                try!(self.word_space(":"));
                self.print_type(&*loc.ty)
            }
        }
    }

    pub fn print_decl(&mut self, decl: &ast::Decl) -> IoResult<()> {
        try!(self.maybe_print_comment(decl.span.lo));
        match decl.node {
            ast::DeclLocal(ref loc) => {
                try!(self.space_if_not_bol());
                try!(self.ibox(indent_unit));
                try!(self.word_nbsp("let"));

                try!(self.ibox(indent_unit));
                try!(self.print_local_decl(&**loc));
                try!(self.end());
                match loc.init {
                    Some(ref init) => {
                        try!(self.nbsp());
                        try!(self.word_space("="));
                        try!(self.print_expr(&**init));
                    }
                    _ => {}
                }
                self.end()
            }
            ast::DeclItem(ref item) => self.print_item(&**item)
        }
    }

    pub fn print_ident(&mut self, ident: ast::Ident) -> IoResult<()> {
        if self.encode_idents_with_hygiene {
            let encoded = ident.encode_with_hygiene();
            try!(word(&mut self.s, encoded.as_slice()))
        } else {
            try!(word(&mut self.s, token::get_ident(ident).get()))
        }
        self.ann.post(self, NodeIdent(&ident))
    }

    pub fn print_uint(&mut self, i: uint) -> IoResult<()> {
        word(&mut self.s, i.to_string().as_slice())
    }

    pub fn print_name(&mut self, name: ast::Name) -> IoResult<()> {
        try!(word(&mut self.s, token::get_name(name).get()));
        self.ann.post(self, NodeName(&name))
    }

    pub fn print_for_decl(&mut self, loc: &ast::Local,
                          coll: &ast::Expr) -> IoResult<()> {
        try!(self.print_local_decl(loc));
        try!(space(&mut self.s));
        try!(self.word_space("in"));
        self.print_expr(coll)
    }

    fn print_path_(&mut self,
                   path: &ast::Path,
                   colons_before_params: bool,
                   opt_bounds: &Option<OwnedSlice<ast::TyParamBound>>)
        -> IoResult<()> {
        try!(self.maybe_print_comment(path.span.lo));
        if path.global {
            try!(word(&mut self.s, "::"));
        }

        let mut first = true;
        for segment in path.segments.iter() {
            if first {
                first = false
            } else {
                try!(word(&mut self.s, "::"))
            }

            try!(self.print_ident(segment.identifier));

            if !segment.lifetimes.is_empty() || !segment.types.is_empty() {
                if colons_before_params {
                    try!(word(&mut self.s, "::"))
                }
                try!(word(&mut self.s, "<"));

                let mut comma = false;
                for lifetime in segment.lifetimes.iter() {
                    if comma {
                        try!(self.word_space(","))
                    }
                    try!(self.print_lifetime(lifetime));
                    comma = true;
                }

                if !segment.types.is_empty() {
                    if comma {
                        try!(self.word_space(","))
                    }
                    try!(self.commasep(
                        Inconsistent,
                        segment.types.as_slice(),
                        |s, ty| s.print_type(&**ty)));
                }

                try!(word(&mut self.s, ">"))
            }
        }

        match *opt_bounds {
            None => Ok(()),
            Some(ref bounds) => self.print_bounds("+", bounds)
        }
    }

    fn print_path(&mut self, path: &ast::Path,
                  colons_before_params: bool) -> IoResult<()> {
        self.print_path_(path, colons_before_params, &None)
    }

    fn print_bounded_path(&mut self, path: &ast::Path,
                          bounds: &Option<OwnedSlice<ast::TyParamBound>>)
        -> IoResult<()> {
        self.print_path_(path, false, bounds)
    }

    pub fn print_pat(&mut self, pat: &ast::Pat) -> IoResult<()> {
        try!(self.maybe_print_comment(pat.span.lo));
        try!(self.ann.pre(self, NodePat(pat)));
        /* Pat isn't normalized, but the beauty of it
         is that it doesn't matter */
        match pat.node {
            ast::PatWild(ast::PatWildSingle) => try!(word(&mut self.s, "_")),
            ast::PatWild(ast::PatWildMulti) => try!(word(&mut self.s, "..")),
            ast::PatIdent(binding_mode, ref path1, ref sub) => {
                match binding_mode {
                    ast::BindByRef(mutbl) => {
                        try!(self.word_nbsp("ref"));
                        try!(self.print_mutability(mutbl));
                    }
                    ast::BindByValue(ast::MutImmutable) => {}
                    ast::BindByValue(ast::MutMutable) => {
                        try!(self.word_nbsp("mut"));
                    }
                }
                try!(self.print_ident(path1.node));
                match *sub {
                    Some(ref p) => {
                        try!(word(&mut self.s, "@"));
                        try!(self.print_pat(&**p));
                    }
                    None => ()
                }
            }
            ast::PatEnum(ref path, ref args_) => {
                try!(self.print_path(path, true));
                match *args_ {
                    None => try!(word(&mut self.s, "(..)")),
                    Some(ref args) => {
                        if !args.is_empty() {
                            try!(self.popen());
                            try!(self.commasep(Inconsistent, args.as_slice(),
                                              |s, p| s.print_pat(&**p)));
                            try!(self.pclose());
                        }
                    }
                }
            }
            ast::PatStruct(ref path, ref fields, etc) => {
                try!(self.print_path(path, true));
                try!(self.nbsp());
                try!(self.word_space("{"));
                try!(self.commasep_cmnt(
                    Consistent, fields.as_slice(),
                    |s, f| {
                        try!(s.cbox(indent_unit));
                        try!(s.print_ident(f.ident));
                        try!(s.word_nbsp(":"));
                        try!(s.print_pat(&*f.pat));
                        s.end()
                    },
                    |f| f.pat.span));
                if etc {
                    if fields.len() != 0u { try!(self.word_space(",")); }
                    try!(word(&mut self.s, ".."));
                }
                try!(space(&mut self.s));
                try!(word(&mut self.s, "}"));
            }
            ast::PatTup(ref elts) => {
                try!(self.popen());
                try!(self.commasep(Inconsistent,
                                   elts.as_slice(),
                                   |s, p| s.print_pat(&**p)));
                if elts.len() == 1 {
                    try!(word(&mut self.s, ","));
                }
                try!(self.pclose());
            }
            ast::PatBox(ref inner) => {
                try!(word(&mut self.s, "box "));
                try!(self.print_pat(&**inner));
            }
            ast::PatRegion(ref inner) => {
                try!(word(&mut self.s, "&"));
                try!(self.print_pat(&**inner));
            }
            ast::PatLit(ref e) => try!(self.print_expr(&**e)),
            ast::PatRange(ref begin, ref end) => {
                try!(self.print_expr(&**begin));
                try!(space(&mut self.s));
                try!(word(&mut self.s, "..."));
                try!(self.print_expr(&**end));
            }
            ast::PatVec(ref before, ref slice, ref after) => {
                try!(word(&mut self.s, "["));
                try!(self.commasep(Inconsistent,
                                   before.as_slice(),
                                   |s, p| s.print_pat(&**p)));
                for p in slice.iter() {
                    if !before.is_empty() { try!(self.word_space(",")); }
                    try!(self.print_pat(&**p));
                    match **p {
                        ast::Pat { node: ast::PatWild(ast::PatWildMulti), .. } => {
                            // this case is handled by print_pat
                        }
                        _ => try!(word(&mut self.s, "..")),
                    }
                    if !after.is_empty() { try!(self.word_space(",")); }
                }
                try!(self.commasep(Inconsistent,
                                   after.as_slice(),
                                   |s, p| s.print_pat(&**p)));
                try!(word(&mut self.s, "]"));
            }
            ast::PatMac(ref m) => try!(self.print_mac(m)),
        }
        self.ann.post(self, NodePat(pat))
    }

    fn print_arm(&mut self, arm: &ast::Arm) -> IoResult<()> {
        // I have no idea why this check is necessary, but here it
        // is :(
        if arm.attrs.is_empty() {
            try!(space(&mut self.s));
        }
        try!(self.cbox(indent_unit));
        try!(self.ibox(0u));
        try!(self.print_outer_attributes(arm.attrs.as_slice()));
        let mut first = true;
        for p in arm.pats.iter() {
            if first {
                first = false;
            } else {
                try!(space(&mut self.s));
                try!(self.word_space("|"));
            }
            try!(self.print_pat(&**p));
        }
        try!(space(&mut self.s));
        match arm.guard {
            Some(ref e) => {
                try!(self.word_space("if"));
                try!(self.print_expr(&**e));
                try!(space(&mut self.s));
            }
            None => ()
        }
        try!(self.word_space("=>"));

        match arm.body.node {
            ast::ExprBlock(ref blk) => {
                // the block will close the pattern's ibox
                try!(self.print_block_unclosed_indent(&**blk,
                                                      indent_unit));
            }
            _ => {
                try!(self.end()); // close the ibox for the pattern
                try!(self.print_expr(&*arm.body));
                try!(word(&mut self.s, ","));
            }
        }
        self.end() // close enclosing cbox
    }

    // Returns whether it printed anything
    fn print_explicit_self(&mut self,
                           explicit_self: &ast::ExplicitSelf_,
                           mutbl: ast::Mutability) -> IoResult<bool> {
        try!(self.print_mutability(mutbl));
        match *explicit_self {
            ast::SelfStatic => { return Ok(false); }
            ast::SelfValue(_) => {
                try!(word(&mut self.s, "self"));
            }
            ast::SelfRegion(ref lt, m, _) => {
                try!(word(&mut self.s, "&"));
                try!(self.print_opt_lifetime(lt));
                try!(self.print_mutability(m));
                try!(word(&mut self.s, "self"));
            }
            ast::SelfExplicit(ref typ, _) => {
                try!(word(&mut self.s, "self"));
                try!(self.word_space(":"));
                try!(self.print_type(&**typ));
            }
        }
        return Ok(true);
    }

    pub fn print_fn(&mut self,
                    decl: &ast::FnDecl,
                    fn_style: Option<ast::FnStyle>,
                    abi: abi::Abi,
                    name: ast::Ident,
                    generics: &ast::Generics,
                    opt_explicit_self: Option<&ast::ExplicitSelf_>,
                    vis: ast::Visibility) -> IoResult<()> {
        try!(self.head(""));
        try!(self.print_fn_header_info(opt_explicit_self, fn_style, abi, vis));
        try!(self.nbsp());
        try!(self.print_ident(name));
        try!(self.print_generics(generics));
        try!(self.print_fn_args_and_ret(decl, opt_explicit_self))
        self.print_where_clause(generics)
    }

    pub fn print_fn_args(&mut self, decl: &ast::FnDecl,
                         opt_explicit_self: Option<&ast::ExplicitSelf_>)
        -> IoResult<()> {
        // It is unfortunate to duplicate the commasep logic, but we want the
        // self type and the args all in the same box.
        try!(self.rbox(0u, Inconsistent));
        let mut first = true;
        for &explicit_self in opt_explicit_self.iter() {
            let m = match explicit_self {
                &ast::SelfStatic => ast::MutImmutable,
                _ => match decl.inputs.get(0).pat.node {
                    ast::PatIdent(ast::BindByValue(m), _, _) => m,
                    _ => ast::MutImmutable
                }
            };
            first = !try!(self.print_explicit_self(explicit_self, m));
        }

        // HACK(eddyb) ignore the separately printed self argument.
        let args = if first {
            decl.inputs.as_slice()
        } else {
            decl.inputs.slice_from(1)
        };

        for arg in args.iter() {
            if first { first = false; } else { try!(self.word_space(",")); }
            try!(self.print_arg(arg));
        }

        self.end()
    }

    pub fn print_fn_args_and_ret(&mut self, decl: &ast::FnDecl,
                                 opt_explicit_self: Option<&ast::ExplicitSelf_>)
        -> IoResult<()> {
        try!(self.popen());
        try!(self.print_fn_args(decl, opt_explicit_self));
        if decl.variadic {
            try!(word(&mut self.s, ", ..."));
        }
        try!(self.pclose());

        try!(self.maybe_print_comment(decl.output.span.lo));
        match decl.output.node {
            ast::TyNil => Ok(()),
            _ => {
                try!(self.space_if_not_bol());
                try!(self.word_space("->"));
                self.print_type(&*decl.output)
            }
        }
    }

    pub fn print_fn_block_args(
            &mut self,
            decl: &ast::FnDecl,
            unboxed_closure_kind: Option<UnboxedClosureKind>)
            -> IoResult<()> {
        try!(word(&mut self.s, "|"));
        match unboxed_closure_kind {
            None => {}
            Some(FnUnboxedClosureKind) => try!(self.word_space("&:")),
            Some(FnMutUnboxedClosureKind) => try!(self.word_space("&mut:")),
            Some(FnOnceUnboxedClosureKind) => try!(self.word_space(":")),
        }
        try!(self.print_fn_args(decl, None));
        try!(word(&mut self.s, "|"));

        match decl.output.node {
            ast::TyInfer => {}
            _ => {
                try!(self.space_if_not_bol());
                try!(self.word_space("->"));
                try!(self.print_type(&*decl.output));
            }
        }

        self.maybe_print_comment(decl.output.span.lo)
    }

    pub fn print_capture_clause(&mut self, capture_clause: ast::CaptureClause)
                                -> IoResult<()> {
        match capture_clause {
            ast::CaptureByValue => self.word_space("move"),
            ast::CaptureByRef => Ok(()),
        }
    }

    pub fn print_proc_args(&mut self, decl: &ast::FnDecl) -> IoResult<()> {
        try!(word(&mut self.s, "proc"));
        try!(word(&mut self.s, "("));
        try!(self.print_fn_args(decl, None));
        try!(word(&mut self.s, ")"));

        match decl.output.node {
            ast::TyInfer => {}
            _ => {
                try!(self.space_if_not_bol());
                try!(self.word_space("->"));
                try!(self.print_type(&*decl.output));
            }
        }

        self.maybe_print_comment(decl.output.span.lo)
    }

    pub fn print_bounds(&mut self,
                        prefix: &str,
                        bounds: &OwnedSlice<ast::TyParamBound>)
                        -> IoResult<()> {
        if !bounds.is_empty() {
            try!(word(&mut self.s, prefix));
            let mut first = true;
            for bound in bounds.iter() {
                try!(self.nbsp());
                if first {
                    first = false;
                } else {
                    try!(self.word_space("+"));
                }

                try!(match *bound {
                    TraitTyParamBound(ref tref) => {
                        self.print_trait_ref(tref)
                    }
                    RegionTyParamBound(ref lt) => {
                        self.print_lifetime(lt)
                    }
                    UnboxedFnTyParamBound(ref unboxed_function_type) => {
                        try!(self.print_path(&unboxed_function_type.path,
                                             false));
                        try!(self.popen());
                        try!(self.print_fn_args(&*unboxed_function_type.decl,
                                                None));
                        try!(self.pclose());
                        self.print_fn_output(&*unboxed_function_type.decl)
                    }
                })
            }
            Ok(())
        } else {
            Ok(())
        }
    }

    pub fn print_lifetime(&mut self,
                          lifetime: &ast::Lifetime)
                          -> IoResult<()>
    {
        self.print_name(lifetime.name)
    }

    pub fn print_lifetime_def(&mut self,
                              lifetime: &ast::LifetimeDef)
                              -> IoResult<()>
    {
        try!(self.print_lifetime(&lifetime.lifetime));
        let mut sep = ":";
        for v in lifetime.bounds.iter() {
            try!(word(&mut self.s, sep));
            try!(self.print_lifetime(v));
            sep = "+";
        }
        Ok(())
    }

    pub fn print_generics(&mut self,
                          generics: &ast::Generics)
                          -> IoResult<()>
    {
        let total = generics.lifetimes.len() + generics.ty_params.len();
        if total == 0 {
            return Ok(());
        }

        try!(word(&mut self.s, "<"));

        let mut ints = Vec::new();
        for i in range(0u, total) {
            ints.push(i);
        }

        try!(self.commasep(Inconsistent, ints.as_slice(), |s, &idx| {
            if idx < generics.lifetimes.len() {
                let lifetime = generics.lifetimes.get(idx);
                s.print_lifetime_def(lifetime)
            } else {
                let idx = idx - generics.lifetimes.len();
                let param = generics.ty_params.get(idx);
                match param.unbound {
                    Some(TraitTyParamBound(ref tref)) => {
                        try!(s.print_trait_ref(tref));
                        try!(s.word_space("?"));
                    }
                    _ => {}
                }
                try!(s.print_ident(param.ident));
                try!(s.print_bounds(":", &param.bounds));
                match param.default {
                    Some(ref default) => {
                        try!(space(&mut s.s));
                        try!(s.word_space("="));
                        s.print_type(&**default)
                    }
                    _ => Ok(())
                }
            }
        }));

        try!(word(&mut self.s, ">"));
        Ok(())
    }

    pub fn print_where_clause(&mut self, generics: &ast::Generics)
                              -> IoResult<()> {
        if generics.where_clause.predicates.len() == 0 {
            return Ok(())
        }

        try!(space(&mut self.s));
        try!(self.word_space("where"));

        for (i, predicate) in generics.where_clause
                                      .predicates
                                      .iter()
                                      .enumerate() {
            if i != 0 {
                try!(self.word_space(","));
            }

            try!(self.print_ident(predicate.ident));
            try!(self.print_bounds(":", &predicate.bounds));
        }

        Ok(())
    }

    pub fn print_meta_item(&mut self, item: &ast::MetaItem) -> IoResult<()> {
        try!(self.ibox(indent_unit));
        match item.node {
            ast::MetaWord(ref name) => {
                try!(word(&mut self.s, name.get()));
            }
            ast::MetaNameValue(ref name, ref value) => {
                try!(self.word_space(name.get()));
                try!(self.word_space("="));
                try!(self.print_literal(value));
            }
            ast::MetaList(ref name, ref items) => {
                try!(word(&mut self.s, name.get()));
                try!(self.popen());
                try!(self.commasep(Consistent,
                                   items.as_slice(),
                                   |s, i| s.print_meta_item(&**i)));
                try!(self.pclose());
            }
        }
        self.end()
    }

    pub fn print_view_path(&mut self, vp: &ast::ViewPath) -> IoResult<()> {
        match vp.node {
            ast::ViewPathSimple(ident, ref path, _) => {
                try!(self.print_path(path, false));

                // FIXME(#6993) can't compare identifiers directly here
                if path.segments.last().unwrap().identifier.name !=
                        ident.name {
                    try!(space(&mut self.s));
                    try!(self.word_space("as"));
                    try!(self.print_ident(ident));
                }

                Ok(())
            }

            ast::ViewPathGlob(ref path, _) => {
                try!(self.print_path(path, false));
                word(&mut self.s, "::*")
            }

            ast::ViewPathList(ref path, ref idents, _) => {
                if path.segments.is_empty() {
                    try!(word(&mut self.s, "{"));
                } else {
                    try!(self.print_path(path, false));
                    try!(word(&mut self.s, "::{"));
                }
                try!(self.commasep(Inconsistent, idents.as_slice(), |s, w| {
                    match w.node {
                        ast::PathListIdent { name, .. } => {
                            s.print_ident(name)
                        },
                        ast::PathListMod { .. } => {
                            word(&mut s.s, "mod")
                        }
                    }
                }));
                word(&mut self.s, "}")
            }
        }
    }

    pub fn print_view_item(&mut self, item: &ast::ViewItem) -> IoResult<()> {
        try!(self.hardbreak_if_not_bol());
        try!(self.maybe_print_comment(item.span.lo));
        try!(self.print_outer_attributes(item.attrs.as_slice()));
        try!(self.print_visibility(item.vis));
        match item.node {
            ast::ViewItemExternCrate(id, ref optional_path, _) => {
                try!(self.head("extern crate"));
                for &(ref p, style) in optional_path.iter() {
                    try!(self.print_string(p.get(), style));
                    try!(space(&mut self.s));
                    try!(word(&mut self.s, "as"));
                    try!(space(&mut self.s));
                }
                try!(self.print_ident(id));
            }

            ast::ViewItemUse(ref vp) => {
                try!(self.head("use"));
                try!(self.print_view_path(&**vp));
            }
        }
        try!(word(&mut self.s, ";"));
        try!(self.end()); // end inner head-block
        self.end() // end outer head-block
    }

    pub fn print_mutability(&mut self,
                            mutbl: ast::Mutability) -> IoResult<()> {
        match mutbl {
            ast::MutMutable => self.word_nbsp("mut"),
            ast::MutImmutable => Ok(()),
        }
    }

    pub fn print_mt(&mut self, mt: &ast::MutTy) -> IoResult<()> {
        try!(self.print_mutability(mt.mutbl));
        self.print_type(&*mt.ty)
    }

    pub fn print_arg(&mut self, input: &ast::Arg) -> IoResult<()> {
        try!(self.ibox(indent_unit));
        match input.ty.node {
            ast::TyInfer => try!(self.print_pat(&*input.pat)),
            _ => {
                match input.pat.node {
                    ast::PatIdent(_, ref path1, _) if
                        path1.node.name ==
                            parse::token::special_idents::invalid.name => {
                        // Do nothing.
                    }
                    _ => {
                        try!(self.print_pat(&*input.pat));
                        try!(word(&mut self.s, ":"));
                        try!(space(&mut self.s));
                    }
                }
                try!(self.print_type(&*input.ty));
            }
        }
        self.end()
    }

    pub fn print_fn_output(&mut self, decl: &ast::FnDecl) -> IoResult<()> {
        match decl.output.node {
            ast::TyNil => Ok(()),
            _ => {
                try!(self.space_if_not_bol());
                try!(self.ibox(indent_unit));
                try!(self.word_space("->"));
                if decl.cf == ast::NoReturn {
                    try!(self.word_nbsp("!"));
                } else {
                    try!(self.print_type(&*decl.output));
                }
                self.end()
            }
        }
    }

    pub fn print_ty_fn(&mut self,
                       opt_abi: Option<abi::Abi>,
                       opt_sigil: Option<char>,
                       fn_style: ast::FnStyle,
                       onceness: ast::Onceness,
                       decl: &ast::FnDecl,
                       id: Option<ast::Ident>,
                       bounds: &OwnedSlice<ast::TyParamBound>,
                       generics: Option<&ast::Generics>,
                       opt_explicit_self: Option<&ast::ExplicitSelf_>,
                       opt_unboxed_closure_kind:
                        Option<ast::UnboxedClosureKind>)
                       -> IoResult<()> {
        try!(self.ibox(indent_unit));

        // Duplicates the logic in `print_fn_header_info()`.  This is because that
        // function prints the sigil in the wrong place.  That should be fixed.
        if opt_sigil == Some('~') && onceness == ast::Once {
            try!(word(&mut self.s, "proc"));
        } else if opt_sigil == Some('&') {
            try!(self.print_fn_style(fn_style));
            try!(self.print_extern_opt_abi(opt_abi));
            try!(self.print_onceness(onceness));
        } else {
            assert!(opt_sigil.is_none());
            try!(self.print_fn_style(fn_style));
            try!(self.print_opt_abi_and_extern_if_nondefault(opt_abi));
            try!(self.print_onceness(onceness));
            if opt_unboxed_closure_kind.is_none() {
                try!(word(&mut self.s, "fn"));
            }
        }

        match id {
            Some(id) => {
                try!(word(&mut self.s, " "));
                try!(self.print_ident(id));
            }
            _ => ()
        }

        match generics { Some(g) => try!(self.print_generics(g)), _ => () }
        try!(zerobreak(&mut self.s));

        if opt_unboxed_closure_kind.is_some() || opt_sigil == Some('&') {
            try!(word(&mut self.s, "|"));
        } else {
            try!(self.popen());
        }

        match opt_unboxed_closure_kind {
            Some(ast::FnUnboxedClosureKind) => {
                try!(word(&mut self.s, "&"));
                try!(self.word_space(":"));
            }
            Some(ast::FnMutUnboxedClosureKind) => {
                try!(word(&mut self.s, "&mut"));
                try!(self.word_space(":"));
            }
            Some(ast::FnOnceUnboxedClosureKind) => {
                try!(self.word_space(":"));
            }
            None => {}
        }

        try!(self.print_fn_args(decl, opt_explicit_self));

        if opt_unboxed_closure_kind.is_some() || opt_sigil == Some('&') {
            try!(word(&mut self.s, "|"));
        } else {
            if decl.variadic {
                try!(word(&mut self.s, ", ..."));
            }
            try!(self.pclose());
        }

        try!(self.print_bounds(":", bounds));

        try!(self.maybe_print_comment(decl.output.span.lo));

        try!(self.print_fn_output(decl));

        match generics {
            Some(generics) => try!(self.print_where_clause(generics)),
            None => {}
        }

        self.end()
    }

    pub fn maybe_print_trailing_comment(&mut self, span: codemap::Span,
                                        next_pos: Option<BytePos>)
        -> IoResult<()> {
        let cm = match self.cm {
            Some(cm) => cm,
            _ => return Ok(())
        };
        match self.next_comment() {
            Some(ref cmnt) => {
                if (*cmnt).style != comments::Trailing { return Ok(()) }
                let span_line = cm.lookup_char_pos(span.hi);
                let comment_line = cm.lookup_char_pos((*cmnt).pos);
                let mut next = (*cmnt).pos + BytePos(1);
                match next_pos { None => (), Some(p) => next = p }
                if span.hi < (*cmnt).pos && (*cmnt).pos < next &&
                    span_line.line == comment_line.line {
                        try!(self.print_comment(cmnt));
                        self.cur_cmnt_and_lit.cur_cmnt += 1u;
                    }
            }
            _ => ()
        }
        Ok(())
    }

    pub fn print_remaining_comments(&mut self) -> IoResult<()> {
        // If there aren't any remaining comments, then we need to manually
        // make sure there is a line break at the end.
        if self.next_comment().is_none() {
            try!(hardbreak(&mut self.s));
        }
        loop {
            match self.next_comment() {
                Some(ref cmnt) => {
                    try!(self.print_comment(cmnt));
                    self.cur_cmnt_and_lit.cur_cmnt += 1u;
                }
                _ => break
            }
        }
        Ok(())
    }

    pub fn print_literal(&mut self, lit: &ast::Lit) -> IoResult<()> {
        try!(self.maybe_print_comment(lit.span.lo));
        match self.next_lit(lit.span.lo) {
            Some(ref ltrl) => {
                return word(&mut self.s, (*ltrl).lit.as_slice());
            }
            _ => ()
        }
        match lit.node {
            ast::LitStr(ref st, style) => self.print_string(st.get(), style),
            ast::LitByte(byte) => {
                let mut res = String::from_str("b'");
                (byte as char).escape_default(|c| res.push_char(c));
                res.push_char('\'');
                word(&mut self.s, res.as_slice())
            }
            ast::LitChar(ch) => {
                let mut res = String::from_str("'");
                ch.escape_default(|c| res.push_char(c));
                res.push_char('\'');
                word(&mut self.s, res.as_slice())
            }
            ast::LitInt(i, t) => {
                match t {
                    ast::SignedIntLit(st, ast::Plus) => {
                        word(&mut self.s,
                             ast_util::int_ty_to_string(st, Some(i as i64)).as_slice())
                    }
                    ast::SignedIntLit(st, ast::Minus) => {
                        word(&mut self.s,
                             ast_util::int_ty_to_string(st, Some(-(i as i64))).as_slice())
                    }
                    ast::UnsignedIntLit(ut) => {
                        word(&mut self.s, ast_util::uint_ty_to_string(ut, Some(i)).as_slice())
                    }
                    ast::UnsuffixedIntLit(ast::Plus) => {
                        word(&mut self.s, format!("{}", i).as_slice())
                    }
                    ast::UnsuffixedIntLit(ast::Minus) => {
                        word(&mut self.s, format!("-{}", i).as_slice())
                    }
                }
            }
            ast::LitFloat(ref f, t) => {
                word(&mut self.s,
                     format!(
                         "{}{}",
                         f.get(),
                         ast_util::float_ty_to_string(t).as_slice()).as_slice())
            }
            ast::LitFloatUnsuffixed(ref f) => word(&mut self.s, f.get()),
            ast::LitNil => word(&mut self.s, "()"),
            ast::LitBool(val) => {
                if val { word(&mut self.s, "true") } else { word(&mut self.s, "false") }
            }
            ast::LitBinary(ref v) => {
                let escaped: String = v.iter().map(|&b| b as char).collect();
                word(&mut self.s, format!("b\"{}\"", escaped.escape_default()).as_slice())
            }
        }
    }

    pub fn next_lit(&mut self, pos: BytePos) -> Option<comments::Literal> {
        match self.literals {
            Some(ref lits) => {
                while self.cur_cmnt_and_lit.cur_lit < lits.len() {
                    let ltrl = (*(*lits).get(self.cur_cmnt_and_lit.cur_lit)).clone();
                    if ltrl.pos > pos { return None; }
                    self.cur_cmnt_and_lit.cur_lit += 1u;
                    if ltrl.pos == pos { return Some(ltrl); }
                }
                None
            }
            _ => None
        }
    }

    pub fn maybe_print_comment(&mut self, pos: BytePos) -> IoResult<()> {
        loop {
            match self.next_comment() {
                Some(ref cmnt) => {
                    if (*cmnt).pos < pos {
                        try!(self.print_comment(cmnt));
                        self.cur_cmnt_and_lit.cur_cmnt += 1u;
                    } else { break; }
                }
                _ => break
            }
        }
        Ok(())
    }

    pub fn print_comment(&mut self,
                         cmnt: &comments::Comment) -> IoResult<()> {
        match cmnt.style {
            comments::Mixed => {
                assert_eq!(cmnt.lines.len(), 1u);
                try!(zerobreak(&mut self.s));
                try!(word(&mut self.s, cmnt.lines.get(0).as_slice()));
                zerobreak(&mut self.s)
            }
            comments::Isolated => {
                try!(self.hardbreak_if_not_bol());
                for line in cmnt.lines.iter() {
                    // Don't print empty lines because they will end up as trailing
                    // whitespace
                    if !line.is_empty() {
                        try!(word(&mut self.s, line.as_slice()));
                    }
                    try!(hardbreak(&mut self.s));
                }
                Ok(())
            }
            comments::Trailing => {
                try!(word(&mut self.s, " "));
                if cmnt.lines.len() == 1u {
                    try!(word(&mut self.s, cmnt.lines.get(0).as_slice()));
                    hardbreak(&mut self.s)
                } else {
                    try!(self.ibox(0u));
                    for line in cmnt.lines.iter() {
                        if !line.is_empty() {
                            try!(word(&mut self.s, line.as_slice()));
                        }
                        try!(hardbreak(&mut self.s));
                    }
                    self.end()
                }
            }
            comments::BlankLine => {
                // We need to do at least one, possibly two hardbreaks.
                let is_semi = match self.s.last_token() {
                    pp::String(s, _) => ";" == s.as_slice(),
                    _ => false
                };
                if is_semi || self.is_begin() || self.is_end() {
                    try!(hardbreak(&mut self.s));
                }
                hardbreak(&mut self.s)
            }
        }
    }

    pub fn print_string(&mut self, st: &str,
                        style: ast::StrStyle) -> IoResult<()> {
        let st = match style {
            ast::CookedStr => {
                (format!("\"{}\"", st.escape_default()))
            }
            ast::RawStr(n) => {
                (format!("r{delim}\"{string}\"{delim}",
                         delim="#".repeat(n),
                         string=st))
            }
        };
        word(&mut self.s, st.as_slice())
    }

    pub fn next_comment(&mut self) -> Option<comments::Comment> {
        match self.comments {
            Some(ref cmnts) => {
                if self.cur_cmnt_and_lit.cur_cmnt < cmnts.len() {
                    Some((*cmnts.get(self.cur_cmnt_and_lit.cur_cmnt)).clone())
                } else {
                    None
                }
            }
            _ => None
        }
    }

    pub fn print_opt_fn_style(&mut self,
                            opt_fn_style: Option<ast::FnStyle>) -> IoResult<()> {
        match opt_fn_style {
            Some(fn_style) => self.print_fn_style(fn_style),
            None => Ok(())
        }
    }

    pub fn print_opt_abi_and_extern_if_nondefault(&mut self,
                                                  opt_abi: Option<abi::Abi>)
        -> IoResult<()> {
        match opt_abi {
            Some(abi::Rust) => Ok(()),
            Some(abi) => {
                try!(self.word_nbsp("extern"));
                self.word_nbsp(abi.to_string().as_slice())
            }
            None => Ok(())
        }
    }

    pub fn print_extern_opt_abi(&mut self,
                                opt_abi: Option<abi::Abi>) -> IoResult<()> {
        match opt_abi {
            Some(abi) => {
                try!(self.word_nbsp("extern"));
                self.word_nbsp(abi.to_string().as_slice())
            }
            None => Ok(())
        }
    }

    pub fn print_fn_header_info(&mut self,
                                _opt_explicit_self: Option<&ast::ExplicitSelf_>,
                                opt_fn_style: Option<ast::FnStyle>,
                                abi: abi::Abi,
                                vis: ast::Visibility) -> IoResult<()> {
        try!(word(&mut self.s, visibility_qualified(vis, "").as_slice()));
        try!(self.print_opt_fn_style(opt_fn_style));

        if abi != abi::Rust {
            try!(self.word_nbsp("extern"));
            try!(self.word_nbsp(abi.to_string().as_slice()));
        }

        word(&mut self.s, "fn")
    }

    pub fn print_fn_style(&mut self, s: ast::FnStyle) -> IoResult<()> {
        match s {
            ast::NormalFn => Ok(()),
            ast::UnsafeFn => self.word_nbsp("unsafe"),
        }
    }

    pub fn print_onceness(&mut self, o: ast::Onceness) -> IoResult<()> {
        match o {
            ast::Once => self.word_nbsp("once"),
            ast::Many => Ok(())
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use ast;
    use ast_util;
    use codemap;
    use parse::token;
    use ptr::P;

    #[test]
    fn test_fun_to_string() {
        let abba_ident = token::str_to_ident("abba");

        let decl = ast::FnDecl {
            inputs: Vec::new(),
            output: P(ast::Ty {id: 0,
                               node: ast::TyNil,
                               span: codemap::DUMMY_SP}),
            cf: ast::Return,
            variadic: false
        };
        let generics = ast_util::empty_generics();
        assert_eq!(&fun_to_string(&decl, ast::NormalFn, abba_ident,
                               None, &generics),
                   &"fn abba()".to_string());
    }

    #[test]
    fn test_variant_to_string() {
        let ident = token::str_to_ident("principal_skinner");

        let var = codemap::respan(codemap::DUMMY_SP, ast::Variant_ {
            name: ident,
            attrs: Vec::new(),
            // making this up as I go.... ?
            kind: ast::TupleVariantKind(Vec::new()),
            id: 0,
            disr_expr: None,
            vis: ast::Public,
        });

        let varstr = variant_to_string(&var);
        assert_eq!(&varstr,&"pub principal_skinner".to_string());
    }
}
