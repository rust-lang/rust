// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use self::AnnNode::*;

use abi;
use ast;
use ast::{MethodImplItem, RegionTyParamBound, TraitTyParamBound, TraitBoundModifier};
use ast::{RequiredMethod, ProvidedMethod, TypeImplItem, TypeTraitItem};
use ast_util;
use attr;
use owned_slice::OwnedSlice;
use attr::{AttrMetaMethods, AttributeMethods};
use codemap::{self, CodeMap, BytePos};
use diagnostic;
use parse::token::{self, BinOpToken, Token, InternedString};
use parse::lexer::comments;
use parse;
use print::pp::{self, break_offset, word, space, zerobreak, hardbreak};
use print::pp::{Breaks, eof};
use print::pp::Breaks::{Consistent, Inconsistent};
use ptr::P;
use std_inject;

use std::{ascii, mem};
use std::old_io::{self, IoResult};
use std::iter;

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

#[derive(Copy)]
pub struct NoAnn;

impl PpAnn for NoAnn {}

#[derive(Copy)]
pub struct CurrentCommentAndLiteral {
    cur_cmnt: usize,
    cur_lit: usize,
}

pub struct State<'a> {
    pub s: pp::Printer,
    cm: Option<&'a CodeMap>,
    comments: Option<Vec<comments::Comment> >,
    literals: Option<Vec<comments::Literal> >,
    cur_cmnt_and_lit: CurrentCommentAndLiteral,
    boxes: Vec<pp::Breaks>,
    ann: &'a (PpAnn+'a),
    encode_idents_with_hygiene: bool,
}

pub fn rust_printer(writer: Box<old_io::Writer+'static>) -> State<'static> {
    static NO_ANN: NoAnn = NoAnn;
    rust_printer_annotated(writer, &NO_ANN)
}

pub fn rust_printer_annotated<'a>(writer: Box<old_io::Writer+'static>,
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

#[allow(non_upper_case_globals)]
pub const indent_unit: usize = 4;

#[allow(non_upper_case_globals)]
pub const default_columns: usize = 78;

/// Requires you to pass an input filename and reader so that
/// it can scan the input text for comments and literals to
/// copy forward.
pub fn print_crate<'a>(cm: &'a CodeMap,
                       span_diagnostic: &diagnostic::SpanHandler,
                       krate: &ast::Crate,
                       filename: String,
                       input: &mut old_io::Reader,
                       out: Box<old_io::Writer+'static>,
                       ann: &'a PpAnn,
                       is_expanded: bool) -> IoResult<()> {
    let mut s = State::new_from_input(cm,
                                      span_diagnostic,
                                      filename,
                                      input,
                                      out,
                                      ann,
                                      is_expanded);
    if is_expanded && std_inject::use_std(krate) {
        // We need to print `#![no_std]` (and its feature gate) so that
        // compiling pretty-printed source won't inject libstd again.
        // However we don't want these attributes in the AST because
        // of the feature gate, so we fake them up here.

        let no_std_meta = attr::mk_word_item(InternedString::new("no_std"));

        // #![feature(no_std)]
        let fake_attr = attr::mk_attr_inner(attr::mk_attr_id(),
                                            attr::mk_list_item(InternedString::new("feature"),
                                                               vec![no_std_meta.clone()]));
        try!(s.print_attribute(&fake_attr));

        // #![no_std]
        let fake_attr = attr::mk_attr_inner(attr::mk_attr_id(), no_std_meta);
        try!(s.print_attribute(&fake_attr));
    }

    try!(s.print_mod(&krate.module, &krate.attrs));
    try!(s.print_remaining_comments());
    eof(&mut s.s)
}

impl<'a> State<'a> {
    pub fn new_from_input(cm: &'a CodeMap,
                          span_diagnostic: &diagnostic::SpanHandler,
                          filename: String,
                          input: &mut old_io::Reader,
                          out: Box<old_io::Writer+'static>,
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
               out: Box<old_io::Writer+'static>,
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

pub fn to_string<F>(f: F) -> String where
    F: FnOnce(&mut State) -> IoResult<()>,
{
    use std::raw::TraitObject;
    let mut s = rust_printer(box Vec::new());
    f(&mut s).unwrap();
    eof(&mut s.s).unwrap();
    let wr = unsafe {
        // FIXME(pcwalton): A nasty function to extract the string from an `old_io::Writer`
        // that we "know" to be a `Vec<u8>` that works around the lack of checked
        // downcasts.
        let obj: &TraitObject = mem::transmute(&s.s.out);
        mem::transmute::<*mut (), &Vec<u8>>(obj.data)
    };
    String::from_utf8(wr.clone()).unwrap()
}

pub fn binop_to_string(op: BinOpToken) -> &'static str {
    match op {
        token::Plus     => "+",
        token::Minus    => "-",
        token::Star     => "*",
        token::Slash    => "/",
        token::Percent  => "%",
        token::Caret    => "^",
        token::And      => "&",
        token::Or       => "|",
        token::Shl      => "<<",
        token::Shr      => ">>",
    }
}

pub fn token_to_string(tok: &Token) -> String {
    match *tok {
        token::Eq                   => "=".to_string(),
        token::Lt                   => "<".to_string(),
        token::Le                   => "<=".to_string(),
        token::EqEq                 => "==".to_string(),
        token::Ne                   => "!=".to_string(),
        token::Ge                   => ">=".to_string(),
        token::Gt                   => ">".to_string(),
        token::Not                  => "!".to_string(),
        token::Tilde                => "~".to_string(),
        token::OrOr                 => "||".to_string(),
        token::AndAnd               => "&&".to_string(),
        token::BinOp(op)            => binop_to_string(op).to_string(),
        token::BinOpEq(op)          => format!("{}=", binop_to_string(op)),

        /* Structural symbols */
        token::At                   => "@".to_string(),
        token::Dot                  => ".".to_string(),
        token::DotDot               => "..".to_string(),
        token::DotDotDot            => "...".to_string(),
        token::Comma                => ",".to_string(),
        token::Semi                 => ";".to_string(),
        token::Colon                => ":".to_string(),
        token::ModSep               => "::".to_string(),
        token::RArrow               => "->".to_string(),
        token::LArrow               => "<-".to_string(),
        token::FatArrow             => "=>".to_string(),
        token::OpenDelim(token::Paren) => "(".to_string(),
        token::CloseDelim(token::Paren) => ")".to_string(),
        token::OpenDelim(token::Bracket) => "[".to_string(),
        token::CloseDelim(token::Bracket) => "]".to_string(),
        token::OpenDelim(token::Brace) => "{".to_string(),
        token::CloseDelim(token::Brace) => "}".to_string(),
        token::Pound                => "#".to_string(),
        token::Dollar               => "$".to_string(),
        token::Question             => "?".to_string(),

        /* Literals */
        token::Literal(lit, suf) => {
            let mut out = match lit {
                token::Byte(b)           => format!("b'{}'", b.as_str()),
                token::Char(c)           => format!("'{}'", c.as_str()),
                token::Float(c)          => c.as_str().to_string(),
                token::Integer(c)        => c.as_str().to_string(),
                token::Str_(s)           => format!("\"{}\"", s.as_str()),
                token::StrRaw(s, n)      => format!("r{delim}\"{string}\"{delim}",
                                                    delim=repeat("#", n),
                                                    string=s.as_str()),
                token::Binary(v)         => format!("b\"{}\"", v.as_str()),
                token::BinaryRaw(s, n)   => format!("br{delim}\"{string}\"{delim}",
                                                    delim=repeat("#", n),
                                                    string=s.as_str()),
            };

            if let Some(s) = suf {
                out.push_str(s.as_str())
            }

            out
        }

        /* Name components */
        token::Ident(s, _)          => token::get_ident(s).to_string(),
        token::Lifetime(s)          => format!("{}", token::get_ident(s)),
        token::Underscore           => "_".to_string(),

        /* Other */
        token::DocComment(s)        => s.as_str().to_string(),
        token::SubstNt(s, _)        => format!("${}", s),
        token::MatchNt(s, t, _, _)  => format!("${}:{}", s, t),
        token::Eof                  => "<eof>".to_string(),
        token::Whitespace           => " ".to_string(),
        token::Comment              => "/* */".to_string(),
        token::Shebang(s)           => format!("/* shebang: {}*/", s.as_str()),

        token::SpecialVarNt(var)    => format!("${}", var.as_str()),

        token::Interpolated(ref nt) => match *nt {
            token::NtExpr(ref e)  => expr_to_string(&**e),
            token::NtMeta(ref e)  => meta_item_to_string(&**e),
            token::NtTy(ref e)    => ty_to_string(&**e),
            token::NtPath(ref e)  => path_to_string(&**e),
            token::NtItem(..)     => "an interpolated item".to_string(),
            token::NtBlock(..)    => "an interpolated block".to_string(),
            token::NtStmt(..)     => "an interpolated statement".to_string(),
            token::NtPat(..)      => "an interpolated pattern".to_string(),
            token::NtIdent(..)    => "an interpolated identifier".to_string(),
            token::NtTT(..)       => "an interpolated tt".to_string(),
        }
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

pub fn bounds_to_string(bounds: &[ast::TyParamBound]) -> String {
    $to_string(|s| s.print_bounds("", bounds))
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
    $to_string(|s| s.print_fn_block_args(p))
}

pub fn path_to_string(p: &ast::Path) -> String {
    $to_string(|s| s.print_path(p, false))
}

pub fn ident_to_string(id: &ast::Ident) -> String {
    $to_string(|s| s.print_ident(*id))
}

pub fn fun_to_string(decl: &ast::FnDecl, unsafety: ast::Unsafety, name: ast::Ident,
                  opt_explicit_self: Option<&ast::ExplicitSelf_>,
                  generics: &ast::Generics) -> String {
    $to_string(|s| {
        try!(s.print_fn(decl, Some(unsafety), abi::Rust,
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
        try!(s.ibox(0));
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
    $to_string(|s| s.print_mac(arg, ::parse::token::Paren))
}

} }

thing_to_string_impls! { to_string }

// FIXME (Issue #16472): the whole `with_hygiene` mod should go away
// after we revise the syntax::ext::quote::ToToken impls to go directly
// to token-trees instea of thing -> string -> token-trees.

pub mod with_hygiene {
    use abi;
    use ast;
    use std::old_io::IoResult;
    use super::indent_unit;

    // This function is the trick that all the rest of the routines
    // hang on.
    pub fn to_string_hyg<F>(f: F) -> String where
        F: FnOnce(&mut super::State) -> IoResult<()>,
    {
        super::to_string(move |s| {
            s.encode_idents_with_hygiene = true;
            f(s)
        })
    }

    thing_to_string_impls! { to_string_hyg }
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
        ast::ExprClosure(..) |
        ast::ExprAssignOp(..) | ast::ExprCast(..) => true,
        _ => false,
    }
}

impl<'a> State<'a> {
    pub fn ibox(&mut self, u: usize) -> IoResult<()> {
        self.boxes.push(pp::Breaks::Inconsistent);
        pp::ibox(&mut self.s, u)
    }

    pub fn end(&mut self) -> IoResult<()> {
        self.boxes.pop().unwrap();
        pp::end(&mut self.s)
    }

    pub fn cbox(&mut self, u: usize) -> IoResult<()> {
        self.boxes.push(pp::Breaks::Consistent);
        pp::cbox(&mut self.s, u)
    }

    // "raw box"
    pub fn rbox(&mut self, u: usize, b: pp::Breaks) -> IoResult<()> {
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
                   indented: usize) -> IoResult<()> {
        self.bclose_maybe_open(span, indented, true)
    }
    pub fn bclose_maybe_open (&mut self, span: codemap::Span,
                              indented: usize, close_box: bool) -> IoResult<()> {
        try!(self.maybe_print_comment(span.hi));
        try!(self.break_offset_if_not_bol(1, -(indented as isize)));
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
        match self.s.last_token() {
            pp::Token::Begin(_) => true,
            _ => false,
        }
    }

    pub fn is_end(&mut self) -> bool {
        match self.s.last_token() {
            pp::Token::End => true,
            _ => false,
        }
    }

    // is this the beginning of a line?
    pub fn is_bol(&mut self) -> bool {
        self.s.last_token().is_eof() || self.s.last_token().is_hardbreak_tok()
    }

    pub fn in_cbox(&self) -> bool {
        match self.boxes.last() {
            Some(&last_box) => last_box == pp::Breaks::Consistent,
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
    pub fn break_offset_if_not_bol(&mut self, n: usize,
                                   off: isize) -> IoResult<()> {
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
        try!(word(&mut self.s, &text[..]));
        try!(space(&mut self.s));
        word(&mut self.s, "*/")
    }

    pub fn commasep<T, F>(&mut self, b: Breaks, elts: &[T], mut op: F) -> IoResult<()> where
        F: FnMut(&mut State, &T) -> IoResult<()>,
    {
        try!(self.rbox(0, b));
        let mut first = true;
        for elt in elts {
            if first { first = false; } else { try!(self.word_space(",")); }
            try!(op(self, elt));
        }
        self.end()
    }


    pub fn commasep_cmnt<T, F, G>(&mut self,
                                  b: Breaks,
                                  elts: &[T],
                                  mut op: F,
                                  mut get_span: G) -> IoResult<()> where
        F: FnMut(&mut State, &T) -> IoResult<()>,
        G: FnMut(&T) -> codemap::Span,
    {
        try!(self.rbox(0, b));
        let len = elts.len();
        let mut i = 0;
        for elt in elts {
            try!(self.maybe_print_comment(get_span(elt).hi));
            try!(op(self, elt));
            i += 1;
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
        for item in &_mod.items {
            try!(self.print_item(&**item));
        }
        Ok(())
    }

    pub fn print_foreign_mod(&mut self, nmod: &ast::ForeignMod,
                             attrs: &[ast::Attribute]) -> IoResult<()> {
        try!(self.print_inner_attributes(attrs));
        for item in &nmod.items {
            try!(self.print_foreign_item(&**item));
        }
        Ok(())
    }

    pub fn print_opt_lifetime(&mut self,
                              lifetime: &Option<ast::Lifetime>) -> IoResult<()> {
        if let Some(l) = *lifetime {
            try!(self.print_lifetime(&l));
            try!(self.nbsp());
        }
        Ok(())
    }

    pub fn print_type(&mut self, ty: &ast::Ty) -> IoResult<()> {
        try!(self.maybe_print_comment(ty.span.lo));
        try!(self.ibox(0));
        match ty.node {
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
                try!(self.commasep(Inconsistent, &elts[..],
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
                try!(self.print_ty_fn(f.abi,
                                      f.unsafety,
                                      &*f.decl,
                                      None,
                                      &generics,
                                      None));
            }
            ast::TyPath(ref path, _) => {
                try!(self.print_path(path, false));
            }
            ast::TyObjectSum(ref ty, ref bounds) => {
                try!(self.print_type(&**ty));
                try!(self.print_bounds("+", &bounds[..]));
            }
            ast::TyPolyTraitRef(ref bounds) => {
                try!(self.print_bounds("", &bounds[..]));
            }
            ast::TyQPath(ref qpath) => {
                try!(self.print_qpath(&**qpath, false))
            }
            ast::TyFixedLengthVec(ref ty, ref v) => {
                try!(word(&mut self.s, "["));
                try!(self.print_type(&**ty));
                try!(word(&mut self.s, "; "));
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
        try!(self.print_outer_attributes(&item.attrs));
        match item.node {
            ast::ForeignItemFn(ref decl, ref generics) => {
                try!(self.print_fn(&**decl, None, abi::Rust, item.ident, generics,
                                   None, item.vis));
                try!(self.end()); // end head-ibox
                try!(word(&mut self.s, ";"));
                self.end() // end the outer fn box
            }
            ast::ForeignItemStatic(ref t, m) => {
                try!(self.head(&visibility_qualified(item.vis,
                                                    "static")));
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
                             -> IoResult<()>
    {
        try!(self.print_outer_attributes(&typedef.attrs));
        try!(self.word_space("type"));
        try!(self.print_ty_param(&typedef.ty_param));
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
        try!(self.print_outer_attributes(&item.attrs));
        try!(self.ann.pre(self, NodeItem(item)));
        match item.node {
            ast::ItemExternCrate(ref optional_path) => {
                try!(self.head(&visibility_qualified(item.vis,
                                                     "extern crate")));
                if let Some((ref p, style)) = *optional_path {
                    try!(self.print_string(p, style));
                    try!(space(&mut self.s));
                    try!(word(&mut self.s, "as"));
                    try!(space(&mut self.s));
                }
                try!(self.print_ident(item.ident));
                try!(word(&mut self.s, ";"));
                try!(self.end()); // end inner head-block
                try!(self.end()); // end outer head-block
            }
            ast::ItemUse(ref vp) => {
                try!(self.head(&visibility_qualified(item.vis,
                                                     "use")));
                try!(self.print_view_path(&**vp));
                try!(word(&mut self.s, ";"));
                try!(self.end()); // end inner head-block
                try!(self.end()); // end outer head-block
            }
            ast::ItemStatic(ref ty, m, ref expr) => {
                try!(self.head(&visibility_qualified(item.vis,
                                                    "static")));
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
            ast::ItemConst(ref ty, ref expr) => {
                try!(self.head(&visibility_qualified(item.vis,
                                                    "const")));
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
            ast::ItemFn(ref decl, unsafety, abi, ref typarams, ref body) => {
                try!(self.print_fn(
                    &**decl,
                    Some(unsafety),
                    abi,
                    item.ident,
                    typarams,
                    None,
                    item.vis
                ));
                try!(word(&mut self.s, " "));
                try!(self.print_block_with_attrs(&**body, &item.attrs));
            }
            ast::ItemMod(ref _mod) => {
                try!(self.head(&visibility_qualified(item.vis,
                                                    "mod")));
                try!(self.print_ident(item.ident));
                try!(self.nbsp());
                try!(self.bopen());
                try!(self.print_mod(_mod, &item.attrs));
                try!(self.bclose(item.span));
            }
            ast::ItemForeignMod(ref nmod) => {
                try!(self.head("extern"));
                try!(self.word_nbsp(&nmod.abi.to_string()));
                try!(self.bopen());
                try!(self.print_foreign_mod(nmod, &item.attrs));
                try!(self.bclose(item.span));
            }
            ast::ItemTy(ref ty, ref params) => {
                try!(self.ibox(indent_unit));
                try!(self.ibox(0));
                try!(self.word_nbsp(&visibility_qualified(item.vis, "type")));
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
                try!(self.head(&visibility_qualified(item.vis,"struct")));
                try!(self.print_struct(&**struct_def, generics, item.ident, item.span));
            }

            ast::ItemImpl(unsafety,
                          polarity,
                          ref generics,
                          ref opt_trait,
                          ref ty,
                          ref impl_items) => {
                try!(self.head(""));
                try!(self.print_visibility(item.vis));
                try!(self.print_unsafety(unsafety));
                try!(self.word_nbsp("impl"));

                if generics.is_parameterized() {
                    try!(self.print_generics(generics));
                    try!(space(&mut self.s));
                }

                match polarity {
                    ast::ImplPolarity::Negative => {
                        try!(word(&mut self.s, "!"));
                    },
                    _ => {}
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
                try!(self.print_inner_attributes(&item.attrs));
                for impl_item in impl_items {
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
            ast::ItemTrait(unsafety, ref generics, ref bounds, ref methods) => {
                try!(self.head(""));
                try!(self.print_visibility(item.vis));
                try!(self.print_unsafety(unsafety));
                try!(self.word_nbsp("trait"));
                try!(self.print_ident(item.ident));
                try!(self.print_generics(generics));
                let mut real_bounds = Vec::with_capacity(bounds.len());
                for b in bounds.iter() {
                    if let TraitTyParamBound(ref ptr, ast::TraitBoundModifier::Maybe) = *b {
                        try!(space(&mut self.s));
                        try!(self.word_space("for ?"));
                        try!(self.print_trait_ref(&ptr.trait_ref));
                    } else {
                        real_bounds.push(b.clone());
                    }
                }
                try!(self.print_bounds(":", &real_bounds[..]));
                try!(self.print_where_clause(generics));
                try!(word(&mut self.s, " "));
                try!(self.bopen());
                for meth in methods {
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
                try!(self.print_tts(&tts[..]));
                try!(self.pclose());
                try!(word(&mut self.s, ";"));
                try!(self.end());
            }
        }
        self.ann.post(self, NodeItem(item))
    }

    fn print_trait_ref(&mut self, t: &ast::TraitRef) -> IoResult<()> {
        self.print_path(&t.path, false)
    }

    fn print_formal_lifetime_list(&mut self, lifetimes: &[ast::LifetimeDef]) -> IoResult<()> {
        if !lifetimes.is_empty() {
            try!(word(&mut self.s, "for<"));
            let mut comma = false;
            for lifetime_def in lifetimes {
                if comma {
                    try!(self.word_space(","))
                }
                try!(self.print_lifetime_def(lifetime_def));
                comma = true;
            }
            try!(word(&mut self.s, ">"));
        }
        Ok(())
    }

    fn print_poly_trait_ref(&mut self, t: &ast::PolyTraitRef) -> IoResult<()> {
        try!(self.print_formal_lifetime_list(&t.bound_lifetimes));
        self.print_trait_ref(&t.trait_ref)
    }

    pub fn print_enum_def(&mut self, enum_definition: &ast::EnumDef,
                          generics: &ast::Generics, ident: ast::Ident,
                          span: codemap::Span,
                          visibility: ast::Visibility) -> IoResult<()> {
        try!(self.head(&visibility_qualified(visibility, "enum")));
        try!(self.print_ident(ident));
        try!(self.print_generics(generics));
        try!(self.print_where_clause(generics));
        try!(space(&mut self.s));
        self.print_variants(&enum_definition.variants, span)
    }

    pub fn print_variants(&mut self,
                          variants: &[P<ast::Variant>],
                          span: codemap::Span) -> IoResult<()> {
        try!(self.bopen());
        for v in variants {
            try!(self.space_if_not_bol());
            try!(self.maybe_print_comment(v.span.lo));
            try!(self.print_outer_attributes(&v.node.attrs));
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
        if ast_util::struct_def_is_tuple_like(struct_def) {
            if !struct_def.fields.is_empty() {
                try!(self.popen());
                try!(self.commasep(
                    Inconsistent, &struct_def.fields,
                    |s, field| {
                        match field.node.kind {
                            ast::NamedField(..) => panic!("unexpected named field"),
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
            try!(self.print_where_clause(generics));
            try!(word(&mut self.s, ";"));
            try!(self.end());
            self.end() // close the outer-box
        } else {
            try!(self.print_where_clause(generics));
            try!(self.nbsp());
            try!(self.bopen());
            try!(self.hardbreak_if_not_bol());

            for field in &struct_def.fields {
                match field.node.kind {
                    ast::UnnamedField(..) => panic!("unexpected unnamed field"),
                    ast::NamedField(ident, visibility) => {
                        try!(self.hardbreak_if_not_bol());
                        try!(self.maybe_print_comment(field.span.lo));
                        try!(self.print_outer_attributes(&field.node.attrs));
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
            ast::TtToken(_, ref tk) => {
                try!(word(&mut self.s, &token_to_string(tk)));
                match *tk {
                    parse::token::DocComment(..) => {
                        hardbreak(&mut self.s)
                    }
                    _ => Ok(())
                }
            }
            ast::TtDelimited(_, ref delimed) => {
                try!(word(&mut self.s, &token_to_string(&delimed.open_token())));
                try!(space(&mut self.s));
                try!(self.print_tts(&delimed.tts));
                try!(space(&mut self.s));
                word(&mut self.s, &token_to_string(&delimed.close_token()))
            },
            ast::TtSequence(_, ref seq) => {
                try!(word(&mut self.s, "$("));
                for tt_elt in &seq.tts {
                    try!(self.print_tt(tt_elt));
                }
                try!(word(&mut self.s, ")"));
                match seq.separator {
                    Some(ref tk) => {
                        try!(word(&mut self.s, &token_to_string(tk)));
                    }
                    None => {},
                }
                match seq.op {
                    ast::ZeroOrMore => word(&mut self.s, "*"),
                    ast::OneOrMore => word(&mut self.s, "+"),
                }
            }
        }
    }

    pub fn print_tts(&mut self, tts: &[ast::TokenTree]) -> IoResult<()> {
        try!(self.ibox(0));
        let mut suppress_space = false;
        for (i, tt) in tts.iter().enumerate() {
            if i != 0 && !suppress_space {
                try!(space(&mut self.s));
            }
            try!(self.print_tt(tt));
            // There should be no space between the module name and the following `::` in paths,
            // otherwise imported macros get re-parsed from crate metadata incorrectly (#20701)
            suppress_space = match tt {
                &ast::TtToken(_, token::Ident(_, token::ModName)) |
                &ast::TtToken(_, token::MatchNt(_, _, _, token::ModName)) |
                &ast::TtToken(_, token::SubstNt(_, token::ModName)) => true,
                _ => false
            }
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
                                       &args[..],
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
        try!(self.print_outer_attributes(&m.attrs));
        try!(self.print_ty_fn(m.abi,
                              m.unsafety,
                              &*m.decl,
                              Some(m.ident),
                              &m.generics,
                              Some(&m.explicit_self.node)));
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
        try!(self.print_outer_attributes(&meth.attrs));
        match meth.node {
            ast::MethDecl(ident,
                          ref generics,
                          abi,
                          ref explicit_self,
                          unsafety,
                          ref decl,
                          ref body,
                          vis) => {
                try!(self.print_fn(&**decl,
                                   Some(unsafety),
                                   abi,
                                   ident,
                                   generics,
                                   Some(&explicit_self.node),
                                   vis));
                try!(word(&mut self.s, " "));
                self.print_block_with_attrs(&**body, &meth.attrs)
            },
            ast::MethMac(codemap::Spanned { node: ast::MacInvocTT(ref pth, ref tts, _),
                                            ..}) => {
                // code copied from ItemMac:
                try!(self.print_path(pth, false));
                try!(word(&mut self.s, "! "));
                try!(self.cbox(indent_unit));
                try!(self.popen());
                try!(self.print_tts(&tts[..]));
                try!(self.pclose());
                try!(word(&mut self.s, ";"));
                self.end()
            }
        }
    }

    pub fn print_outer_attributes(&mut self,
                                  attrs: &[ast::Attribute]) -> IoResult<()> {
        let mut count = 0;
        for attr in attrs {
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
        let mut count = 0;
        for attr in attrs {
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
            word(&mut self.s, &attr.value_str().unwrap())
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
            ast::StmtMac(ref mac, style) => {
                try!(self.space_if_not_bol());
                let delim = match style {
                    ast::MacStmtWithBraces => token::Brace,
                    _ => token::Paren
                };
                try!(self.print_mac(&**mac, delim));
                match style {
                    ast::MacStmtWithBraces => {}
                    _ => try!(word(&mut self.s, ";")),
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
                                       indented: usize) -> IoResult<()> {
        self.print_block_maybe_unclosed(blk, indented, &[], false)
    }

    pub fn print_block_with_attrs(&mut self,
                                  blk: &ast::Block,
                                  attrs: &[ast::Attribute]) -> IoResult<()> {
        self.print_block_maybe_unclosed(blk, indent_unit, attrs, true)
    }

    pub fn print_block_maybe_unclosed(&mut self,
                                      blk: &ast::Block,
                                      indented: usize,
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

        for st in &blk.stmts {
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
                        try!(self.cbox(indent_unit - 1));
                        try!(self.ibox(0));
                        try!(word(&mut self.s, " else if "));
                        try!(self.print_expr(&**i));
                        try!(space(&mut self.s));
                        try!(self.print_block(&**then));
                        self.print_else(e.as_ref().map(|e| &**e))
                    }
                    // "another else-if-let"
                    ast::ExprIfLet(ref pat, ref expr, ref then, ref e) => {
                        try!(self.cbox(indent_unit - 1));
                        try!(self.ibox(0));
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
                        try!(self.cbox(indent_unit - 1));
                        try!(self.ibox(0));
                        try!(word(&mut self.s, " else "));
                        self.print_block(&**b)
                    }
                    // BLEAH, constraints would be great here
                    _ => {
                        panic!("print_if saw if with weird alternative");
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

    pub fn print_mac(&mut self, m: &ast::Mac, delim: token::DelimToken)
                     -> IoResult<()> {
        match m.node {
            // I think it's reasonable to hide the ctxt here:
            ast::MacInvocTT(ref pth, ref tts, _) => {
                try!(self.print_path(pth, false));
                try!(word(&mut self.s, "!"));
                match delim {
                    token::Paren => try!(self.popen()),
                    token::Bracket => try!(word(&mut self.s, "[")),
                    token::Brace => try!(self.bopen()),
                }
                try!(self.print_tts(tts));
                match delim {
                    token::Paren => self.pclose(),
                    token::Bracket => word(&mut self.s, "]"),
                    token::Brace => self.bclose(m.span),
                }
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

    fn print_expr_box(&mut self,
                      place: &Option<P<ast::Expr>>,
                      expr: &ast::Expr) -> IoResult<()> {
        try!(word(&mut self.s, "box"));
        try!(word(&mut self.s, "("));
        try!(place.as_ref().map_or(Ok(()), |e|self.print_expr(&**e)));
        try!(self.word_space(")"));
        self.print_expr(expr)
    }

    fn print_expr_vec(&mut self, exprs: &[P<ast::Expr>]) -> IoResult<()> {
        try!(self.ibox(indent_unit));
        try!(word(&mut self.s, "["));
        try!(self.commasep_exprs(Inconsistent, &exprs[..]));
        try!(word(&mut self.s, "]"));
        self.end()
    }

    fn print_expr_repeat(&mut self,
                         element: &ast::Expr,
                         count: &ast::Expr) -> IoResult<()> {
        try!(self.ibox(indent_unit));
        try!(word(&mut self.s, "["));
        try!(self.print_expr(element));
        try!(self.word_space(";"));
        try!(self.print_expr(count));
        try!(word(&mut self.s, "]"));
        self.end()
    }

    fn print_expr_struct(&mut self,
                         path: &ast::Path,
                         fields: &[ast::Field],
                         wth: &Option<P<ast::Expr>>) -> IoResult<()> {
        try!(self.print_path(path, true));
        if !(fields.is_empty() && wth.is_none()) {
            try!(word(&mut self.s, "{"));
            try!(self.commasep_cmnt(
                Consistent,
                &fields[..],
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
                _ => try!(word(&mut self.s, ",")),
            }
            try!(word(&mut self.s, "}"));
        }
        Ok(())
    }

    fn print_expr_tup(&mut self, exprs: &[P<ast::Expr>]) -> IoResult<()> {
        try!(self.popen());
        try!(self.commasep_exprs(Inconsistent, &exprs[..]));
        if exprs.len() == 1 {
            try!(word(&mut self.s, ","));
        }
        self.pclose()
    }

    fn print_expr_call(&mut self,
                       func: &ast::Expr,
                       args: &[P<ast::Expr>]) -> IoResult<()> {
        try!(self.print_expr_maybe_paren(func));
        self.print_call_post(args)
    }

    fn print_expr_method_call(&mut self,
                              ident: ast::SpannedIdent,
                              tys: &[P<ast::Ty>],
                              args: &[P<ast::Expr>]) -> IoResult<()> {
        let base_args = &args[1..];
        try!(self.print_expr(&*args[0]));
        try!(word(&mut self.s, "."));
        try!(self.print_ident(ident.node));
        if tys.len() > 0 {
            try!(word(&mut self.s, "::<"));
            try!(self.commasep(Inconsistent, tys,
                               |s, ty| s.print_type(&**ty)));
            try!(word(&mut self.s, ">"));
        }
        self.print_call_post(base_args)
    }

    fn print_expr_binary(&mut self,
                         op: ast::BinOp,
                         lhs: &ast::Expr,
                         rhs: &ast::Expr) -> IoResult<()> {
        try!(self.print_expr(lhs));
        try!(space(&mut self.s));
        try!(self.word_space(ast_util::binop_to_string(op.node)));
        self.print_expr(rhs)
    }

    fn print_expr_unary(&mut self,
                        op: ast::UnOp,
                        expr: &ast::Expr) -> IoResult<()> {
        try!(word(&mut self.s, ast_util::unop_to_string(op)));
        self.print_expr_maybe_paren(expr)
    }

    fn print_expr_addr_of(&mut self,
                          mutability: ast::Mutability,
                          expr: &ast::Expr) -> IoResult<()> {
        try!(word(&mut self.s, "&"));
        try!(self.print_mutability(mutability));
        self.print_expr_maybe_paren(expr)
    }

    pub fn print_expr(&mut self, expr: &ast::Expr) -> IoResult<()> {
        try!(self.maybe_print_comment(expr.span.lo));
        try!(self.ibox(indent_unit));
        try!(self.ann.pre(self, NodeExpr(expr)));
        match expr.node {
            ast::ExprBox(ref place, ref expr) => {
                try!(self.print_expr_box(place, &**expr));
            }
            ast::ExprVec(ref exprs) => {
                try!(self.print_expr_vec(&exprs[..]));
            }
            ast::ExprRepeat(ref element, ref count) => {
                try!(self.print_expr_repeat(&**element, &**count));
            }
            ast::ExprStruct(ref path, ref fields, ref wth) => {
                try!(self.print_expr_struct(path, &fields[..], wth));
            }
            ast::ExprTup(ref exprs) => {
                try!(self.print_expr_tup(&exprs[..]));
            }
            ast::ExprCall(ref func, ref args) => {
                try!(self.print_expr_call(&**func, &args[..]));
            }
            ast::ExprMethodCall(ident, ref tys, ref args) => {
                try!(self.print_expr_method_call(ident, &tys[..], &args[..]));
            }
            ast::ExprBinary(op, ref lhs, ref rhs) => {
                try!(self.print_expr_binary(op, &**lhs, &**rhs));
            }
            ast::ExprUnary(op, ref expr) => {
                try!(self.print_expr_unary(op, &**expr));
            }
            ast::ExprAddrOf(m, ref expr) => {
                try!(self.print_expr_addr_of(m, &**expr));
            }
            ast::ExprLit(ref lit) => {
                try!(self.print_literal(&**lit));
            }
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
                if let Some(ident) = opt_ident {
                    try!(self.print_ident(ident));
                    try!(self.word_space(":"));
                }
                try!(self.head("while"));
                try!(self.print_expr(&**test));
                try!(space(&mut self.s));
                try!(self.print_block(&**blk));
            }
            ast::ExprWhileLet(ref pat, ref expr, ref blk, opt_ident) => {
                if let Some(ident) = opt_ident {
                    try!(self.print_ident(ident));
                    try!(self.word_space(":"));
                }
                try!(self.head("while let"));
                try!(self.print_pat(&**pat));
                try!(space(&mut self.s));
                try!(self.word_space("="));
                try!(self.print_expr(&**expr));
                try!(space(&mut self.s));
                try!(self.print_block(&**blk));
            }
            ast::ExprForLoop(ref pat, ref iter, ref blk, opt_ident) => {
                if let Some(ident) = opt_ident {
                    try!(self.print_ident(ident));
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
                if let Some(ident) = opt_ident {
                    try!(self.print_ident(ident));
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
                for arm in arms {
                    try!(self.print_arm(arm));
                }
                try!(self.bclose_(expr.span, indent_unit));
            }
            ast::ExprClosure(capture_clause, ref decl, ref body) => {
                try!(self.print_capture_clause(capture_clause));

                try!(self.print_fn_block_args(&**decl));
                try!(space(&mut self.s));

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
            ast::ExprBlock(ref blk) => {
                // containing cbox, will be closed by print-block at }
                try!(self.cbox(indent_unit));
                // head-box, will be closed by print-block after {
                try!(self.ibox(0));
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
                try!(word(&mut self.s, ast_util::binop_to_string(op.node)));
                try!(self.word_space("="));
                try!(self.print_expr(&**rhs));
            }
            ast::ExprField(ref expr, id) => {
                try!(self.print_expr(&**expr));
                try!(word(&mut self.s, "."));
                try!(self.print_ident(id.node));
            }
            ast::ExprTupField(ref expr, id) => {
                try!(self.print_expr(&**expr));
                try!(word(&mut self.s, "."));
                try!(self.print_usize(id.node));
            }
            ast::ExprIndex(ref expr, ref index) => {
                try!(self.print_expr(&**expr));
                try!(word(&mut self.s, "["));
                try!(self.print_expr(&**index));
                try!(word(&mut self.s, "]"));
            }
            ast::ExprRange(ref start, ref end) => {
                if let &Some(ref e) = start {
                    try!(self.print_expr(&**e));
                }
                try!(word(&mut self.s, ".."));
                if let &Some(ref e) = end {
                    try!(self.print_expr(&**e));
                }
            }
            ast::ExprPath(ref path) => try!(self.print_path(path, true)),
            ast::ExprQPath(ref qpath) => try!(self.print_qpath(&**qpath, true)),
            ast::ExprBreak(opt_ident) => {
                try!(word(&mut self.s, "break"));
                try!(space(&mut self.s));
                if let Some(ident) = opt_ident {
                    try!(self.print_ident(ident));
                    try!(space(&mut self.s));
                }
            }
            ast::ExprAgain(opt_ident) => {
                try!(word(&mut self.s, "continue"));
                try!(space(&mut self.s));
                if let Some(ident) = opt_ident {
                    try!(self.print_ident(ident));
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
                try!(word(&mut self.s, "asm!"));
                try!(self.popen());
                try!(self.print_string(&a.asm, a.asm_str_style));
                try!(self.word_space(":"));

                try!(self.commasep(Inconsistent, &a.outputs,
                                   |s, &(ref co, ref o, is_rw)| {
                    match co.slice_shift_char() {
                        Some(('=', operand)) if is_rw => {
                            try!(s.print_string(&format!("+{}", operand),
                                                ast::CookedStr))
                        }
                        _ => try!(s.print_string(&co, ast::CookedStr))
                    }
                    try!(s.popen());
                    try!(s.print_expr(&**o));
                    try!(s.pclose());
                    Ok(())
                }));
                try!(space(&mut self.s));
                try!(self.word_space(":"));

                try!(self.commasep(Inconsistent, &a.inputs,
                                   |s, &(ref co, ref o)| {
                    try!(s.print_string(&co, ast::CookedStr));
                    try!(s.popen());
                    try!(s.print_expr(&**o));
                    try!(s.pclose());
                    Ok(())
                }));
                try!(space(&mut self.s));
                try!(self.word_space(":"));

                try!(self.commasep(Inconsistent, &a.clobbers,
                                   |s, co| {
                    try!(s.print_string(&co, ast::CookedStr));
                    Ok(())
                }));

                let mut options = vec!();
                if a.volatile {
                    options.push("volatile");
                }
                if a.alignstack {
                    options.push("alignstack");
                }
                if a.dialect == ast::AsmDialect::AsmIntel {
                    options.push("intel");
                }

                if options.len() > 0 {
                    try!(space(&mut self.s));
                    try!(self.word_space(":"));
                    try!(self.commasep(Inconsistent, &*options,
                                       |s, &co| {
                        try!(s.print_string(co, ast::CookedStr));
                        Ok(())
                    }));
                }

                try!(self.pclose());
            }
            ast::ExprMac(ref m) => try!(self.print_mac(m, token::Paren)),
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
        if let Some(ref ty) = loc.ty {
            try!(self.word_space(":"));
            try!(self.print_type(&**ty));
        }
        Ok(())
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
                if let Some(ref init) = loc.init {
                    try!(self.nbsp());
                    try!(self.word_space("="));
                    try!(self.print_expr(&**init));
                }
                self.end()
            }
            ast::DeclItem(ref item) => self.print_item(&**item)
        }
    }

    pub fn print_ident(&mut self, ident: ast::Ident) -> IoResult<()> {
        if self.encode_idents_with_hygiene {
            let encoded = ident.encode_with_hygiene();
            try!(word(&mut self.s, &encoded[..]))
        } else {
            try!(word(&mut self.s, &token::get_ident(ident)))
        }
        self.ann.post(self, NodeIdent(&ident))
    }

    pub fn print_usize(&mut self, i: usize) -> IoResult<()> {
        word(&mut self.s, &i.to_string())
    }

    pub fn print_name(&mut self, name: ast::Name) -> IoResult<()> {
        try!(word(&mut self.s, &token::get_name(name)));
        self.ann.post(self, NodeName(&name))
    }

    pub fn print_for_decl(&mut self, loc: &ast::Local,
                          coll: &ast::Expr) -> IoResult<()> {
        try!(self.print_local_decl(loc));
        try!(space(&mut self.s));
        try!(self.word_space("in"));
        self.print_expr(coll)
    }

    fn print_path(&mut self,
                  path: &ast::Path,
                  colons_before_params: bool)
                  -> IoResult<()>
    {
        try!(self.maybe_print_comment(path.span.lo));
        if path.global {
            try!(word(&mut self.s, "::"));
        }

        let mut first = true;
        for segment in &path.segments {
            if first {
                first = false
            } else {
                try!(word(&mut self.s, "::"))
            }

            try!(self.print_ident(segment.identifier));

            try!(self.print_path_parameters(&segment.parameters, colons_before_params));
        }

        Ok(())
    }

    fn print_qpath(&mut self,
                   qpath: &ast::QPath,
                   colons_before_params: bool)
                   -> IoResult<()>
    {
        try!(word(&mut self.s, "<"));
        try!(self.print_type(&*qpath.self_type));
        try!(space(&mut self.s));
        try!(self.word_space("as"));
        try!(self.print_trait_ref(&*qpath.trait_ref));
        try!(word(&mut self.s, ">"));
        try!(word(&mut self.s, "::"));
        try!(self.print_ident(qpath.item_path.identifier));
        self.print_path_parameters(&qpath.item_path.parameters, colons_before_params)
    }

    fn print_path_parameters(&mut self,
                             parameters: &ast::PathParameters,
                             colons_before_params: bool)
                             -> IoResult<()>
    {
        if parameters.is_empty() {
            return Ok(());
        }

        if colons_before_params {
            try!(word(&mut self.s, "::"))
        }

        match *parameters {
            ast::AngleBracketedParameters(ref data) => {
                try!(word(&mut self.s, "<"));

                let mut comma = false;
                for lifetime in &data.lifetimes {
                    if comma {
                        try!(self.word_space(","))
                    }
                    try!(self.print_lifetime(lifetime));
                    comma = true;
                }

                if !data.types.is_empty() {
                    if comma {
                        try!(self.word_space(","))
                    }
                    try!(self.commasep(
                        Inconsistent,
                        &data.types,
                        |s, ty| s.print_type(&**ty)));
                        comma = true;
                }

                for binding in &*data.bindings {
                    if comma {
                        try!(self.word_space(","))
                    }
                    try!(self.print_ident(binding.ident));
                    try!(space(&mut self.s));
                    try!(self.word_space("="));
                    try!(self.print_type(&*binding.ty));
                    comma = true;
                }

                try!(word(&mut self.s, ">"))
            }

            ast::ParenthesizedParameters(ref data) => {
                try!(word(&mut self.s, "("));
                try!(self.commasep(
                    Inconsistent,
                    &data.inputs,
                    |s, ty| s.print_type(&**ty)));
                try!(word(&mut self.s, ")"));

                match data.output {
                    None => { }
                    Some(ref ty) => {
                        try!(self.space_if_not_bol());
                        try!(self.word_space("->"));
                        try!(self.print_type(&**ty));
                    }
                }
            }
        }

        Ok(())
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
                            try!(self.commasep(Inconsistent, &args[..],
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
                    Consistent, &fields[..],
                    |s, f| {
                        try!(s.cbox(indent_unit));
                        if !f.node.is_shorthand {
                            try!(s.print_ident(f.node.ident));
                            try!(s.word_nbsp(":"));
                        }
                        try!(s.print_pat(&*f.node.pat));
                        s.end()
                    },
                    |f| f.node.pat.span));
                if etc {
                    if fields.len() != 0 { try!(self.word_space(",")); }
                    try!(word(&mut self.s, ".."));
                }
                try!(space(&mut self.s));
                try!(word(&mut self.s, "}"));
            }
            ast::PatTup(ref elts) => {
                try!(self.popen());
                try!(self.commasep(Inconsistent,
                                   &elts[..],
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
            ast::PatRegion(ref inner, mutbl) => {
                try!(word(&mut self.s, "&"));
                if mutbl == ast::MutMutable {
                    try!(word(&mut self.s, "mut "));
                }
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
                                   &before[..],
                                   |s, p| s.print_pat(&**p)));
                if let Some(ref p) = *slice {
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
                                   &after[..],
                                   |s, p| s.print_pat(&**p)));
                try!(word(&mut self.s, "]"));
            }
            ast::PatMac(ref m) => try!(self.print_mac(m, token::Paren)),
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
        try!(self.ibox(0));
        try!(self.print_outer_attributes(&arm.attrs));
        let mut first = true;
        for p in &arm.pats {
            if first {
                first = false;
            } else {
                try!(space(&mut self.s));
                try!(self.word_space("|"));
            }
            try!(self.print_pat(&**p));
        }
        try!(space(&mut self.s));
        if let Some(ref e) = arm.guard {
            try!(self.word_space("if"));
            try!(self.print_expr(&**e));
            try!(space(&mut self.s));
        }
        try!(self.word_space("=>"));

        match arm.body.node {
            ast::ExprBlock(ref blk) => {
                // the block will close the pattern's ibox
                try!(self.print_block_unclosed_indent(&**blk, indent_unit));

                // If it is a user-provided unsafe block, print a comma after it
                if let ast::UnsafeBlock(ast::UserProvided) = blk.rules {
                    try!(word(&mut self.s, ","));
                }
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
                    unsafety: Option<ast::Unsafety>,
                    abi: abi::Abi,
                    name: ast::Ident,
                    generics: &ast::Generics,
                    opt_explicit_self: Option<&ast::ExplicitSelf_>,
                    vis: ast::Visibility) -> IoResult<()> {
        try!(self.head(""));
        try!(self.print_fn_header_info(unsafety, abi, vis));
        try!(self.nbsp());
        try!(self.print_ident(name));
        try!(self.print_generics(generics));
        try!(self.print_fn_args_and_ret(decl, opt_explicit_self));
        self.print_where_clause(generics)
    }

    pub fn print_fn_args(&mut self, decl: &ast::FnDecl,
                         opt_explicit_self: Option<&ast::ExplicitSelf_>)
        -> IoResult<()> {
        // It is unfortunate to duplicate the commasep logic, but we want the
        // self type and the args all in the same box.
        try!(self.rbox(0, Inconsistent));
        let mut first = true;
        if let Some(explicit_self) = opt_explicit_self {
            let m = match explicit_self {
                &ast::SelfStatic => ast::MutImmutable,
                _ => match decl.inputs[0].pat.node {
                    ast::PatIdent(ast::BindByValue(m), _, _) => m,
                    _ => ast::MutImmutable
                }
            };
            first = !try!(self.print_explicit_self(explicit_self, m));
        }

        // HACK(eddyb) ignore the separately printed self argument.
        let args = if first {
            &decl.inputs[]
        } else {
            &decl.inputs[1..]
        };

        for arg in args {
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

        self.print_fn_output(decl)
    }

    pub fn print_fn_block_args(
            &mut self,
            decl: &ast::FnDecl)
            -> IoResult<()> {
        try!(word(&mut self.s, "|"));
        try!(self.print_fn_args(decl, None));
        try!(word(&mut self.s, "|"));

        if let ast::DefaultReturn(..) = decl.output {
            return Ok(());
        }

        try!(self.space_if_not_bol());
        try!(self.word_space("->"));
        match decl.output {
            ast::Return(ref ty) => {
                try!(self.print_type(&**ty));
                self.maybe_print_comment(ty.span.lo)
            }
            ast::DefaultReturn(..) => unreachable!(),
            ast::NoReturn(span) => {
                try!(self.word_nbsp("!"));
                self.maybe_print_comment(span.lo)
            }
        }
    }

    pub fn print_capture_clause(&mut self, capture_clause: ast::CaptureClause)
                                -> IoResult<()> {
        match capture_clause {
            ast::CaptureByValue => self.word_space("move"),
            ast::CaptureByRef => Ok(()),
        }
    }

    pub fn print_bounds(&mut self,
                        prefix: &str,
                        bounds: &[ast::TyParamBound])
                        -> IoResult<()> {
        if !bounds.is_empty() {
            try!(word(&mut self.s, prefix));
            let mut first = true;
            for bound in bounds {
                try!(self.nbsp());
                if first {
                    first = false;
                } else {
                    try!(self.word_space("+"));
                }

                try!(match *bound {
                    TraitTyParamBound(ref tref, TraitBoundModifier::None) => {
                        self.print_poly_trait_ref(tref)
                    }
                    TraitTyParamBound(ref tref, TraitBoundModifier::Maybe) => {
                        try!(word(&mut self.s, "?"));
                        self.print_poly_trait_ref(tref)
                    }
                    RegionTyParamBound(ref lt) => {
                        self.print_lifetime(lt)
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
        for v in &lifetime.bounds {
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
        for i in 0..total {
            ints.push(i);
        }

        try!(self.commasep(Inconsistent, &ints[..], |s, &idx| {
            if idx < generics.lifetimes.len() {
                let lifetime = &generics.lifetimes[idx];
                s.print_lifetime_def(lifetime)
            } else {
                let idx = idx - generics.lifetimes.len();
                let param = &generics.ty_params[idx];
                s.print_ty_param(param)
            }
        }));

        try!(word(&mut self.s, ">"));
        Ok(())
    }

    pub fn print_ty_param(&mut self, param: &ast::TyParam) -> IoResult<()> {
        try!(self.print_ident(param.ident));
        try!(self.print_bounds(":", &param.bounds));
        match param.default {
            Some(ref default) => {
                try!(space(&mut self.s));
                try!(self.word_space("="));
                self.print_type(&**default)
            }
            _ => Ok(())
        }
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

            match predicate {
                &ast::WherePredicate::BoundPredicate(ast::WhereBoundPredicate{ref bound_lifetimes,
                                                                              ref bounded_ty,
                                                                              ref bounds,
                                                                              ..}) => {
                    try!(self.print_formal_lifetime_list(bound_lifetimes));
                    try!(self.print_type(&**bounded_ty));
                    try!(self.print_bounds(":", bounds));
                }
                &ast::WherePredicate::RegionPredicate(ast::WhereRegionPredicate{ref lifetime,
                                                                                ref bounds,
                                                                                ..}) => {
                    try!(self.print_lifetime(lifetime));
                    try!(word(&mut self.s, ":"));

                    for (i, bound) in bounds.iter().enumerate() {
                        try!(self.print_lifetime(bound));

                        if i != 0 {
                            try!(word(&mut self.s, ":"));
                        }
                    }
                }
                &ast::WherePredicate::EqPredicate(ast::WhereEqPredicate{ref path, ref ty, ..}) => {
                    try!(self.print_path(path, false));
                    try!(space(&mut self.s));
                    try!(self.word_space("="));
                    try!(self.print_type(&**ty));
                }
            }
        }

        Ok(())
    }

    pub fn print_meta_item(&mut self, item: &ast::MetaItem) -> IoResult<()> {
        try!(self.ibox(indent_unit));
        match item.node {
            ast::MetaWord(ref name) => {
                try!(word(&mut self.s, &name));
            }
            ast::MetaNameValue(ref name, ref value) => {
                try!(self.word_space(&name[..]));
                try!(self.word_space("="));
                try!(self.print_literal(value));
            }
            ast::MetaList(ref name, ref items) => {
                try!(word(&mut self.s, &name));
                try!(self.popen());
                try!(self.commasep(Consistent,
                                   &items[..],
                                   |s, i| s.print_meta_item(&**i)));
                try!(self.pclose());
            }
        }
        self.end()
    }

    pub fn print_view_path(&mut self, vp: &ast::ViewPath) -> IoResult<()> {
        match vp.node {
            ast::ViewPathSimple(ident, ref path) => {
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

            ast::ViewPathGlob(ref path) => {
                try!(self.print_path(path, false));
                word(&mut self.s, "::*")
            }

            ast::ViewPathList(ref path, ref idents) => {
                if path.segments.is_empty() {
                    try!(word(&mut self.s, "{"));
                } else {
                    try!(self.print_path(path, false));
                    try!(word(&mut self.s, "::{"));
                }
                try!(self.commasep(Inconsistent, &idents[..], |s, w| {
                    match w.node {
                        ast::PathListIdent { name, .. } => {
                            s.print_ident(name)
                        },
                        ast::PathListMod { .. } => {
                            word(&mut s.s, "self")
                        }
                    }
                }));
                word(&mut self.s, "}")
            }
        }
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
        if let ast::DefaultReturn(..) = decl.output {
            return Ok(());
        }

        try!(self.space_if_not_bol());
        try!(self.ibox(indent_unit));
        try!(self.word_space("->"));
        match decl.output {
            ast::NoReturn(_) =>
                try!(self.word_nbsp("!")),
            ast::DefaultReturn(..) => unreachable!(),
            ast::Return(ref ty) =>
                try!(self.print_type(&**ty))
        }
        try!(self.end());

        match decl.output {
            ast::Return(ref output) => self.maybe_print_comment(output.span.lo),
            _ => Ok(())
        }
    }

    pub fn print_ty_fn(&mut self,
                       abi: abi::Abi,
                       unsafety: ast::Unsafety,
                       decl: &ast::FnDecl,
                       id: Option<ast::Ident>,
                       generics: &ast::Generics,
                       opt_explicit_self: Option<&ast::ExplicitSelf_>)
                       -> IoResult<()> {
        try!(self.ibox(indent_unit));
        try!(self.print_fn_header_info(Some(unsafety), abi, ast::Inherited));

        match id {
            Some(id) => {
                try!(word(&mut self.s, " "));
                try!(self.print_ident(id));
            }
            _ => ()
        }

        try!(self.print_generics(generics));
        try!(zerobreak(&mut self.s));
        try!(self.print_fn_args_and_ret(decl, opt_explicit_self));
        try!(self.print_where_clause(generics));
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
                        self.cur_cmnt_and_lit.cur_cmnt += 1;
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
                    self.cur_cmnt_and_lit.cur_cmnt += 1;
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
                return word(&mut self.s, &(*ltrl).lit);
            }
            _ => ()
        }
        match lit.node {
            ast::LitStr(ref st, style) => self.print_string(&st, style),
            ast::LitByte(byte) => {
                let mut res = String::from_str("b'");
                res.extend(ascii::escape_default(byte).map(|c| c as char));
                res.push('\'');
                word(&mut self.s, &res[..])
            }
            ast::LitChar(ch) => {
                let mut res = String::from_str("'");
                res.extend(ch.escape_default());
                res.push('\'');
                word(&mut self.s, &res[..])
            }
            ast::LitInt(i, t) => {
                match t {
                    ast::SignedIntLit(st, ast::Plus) => {
                        word(&mut self.s,
                             &ast_util::int_ty_to_string(st, Some(i as i64)))
                    }
                    ast::SignedIntLit(st, ast::Minus) => {
                        let istr = ast_util::int_ty_to_string(st, Some(-(i as i64)));
                        word(&mut self.s,
                             &format!("-{}", istr))
                    }
                    ast::UnsignedIntLit(ut) => {
                        word(&mut self.s, &ast_util::uint_ty_to_string(ut, Some(i)))
                    }
                    ast::UnsuffixedIntLit(ast::Plus) => {
                        word(&mut self.s, &format!("{}", i))
                    }
                    ast::UnsuffixedIntLit(ast::Minus) => {
                        word(&mut self.s, &format!("-{}", i))
                    }
                }
            }
            ast::LitFloat(ref f, t) => {
                word(&mut self.s,
                     &format!(
                         "{}{}",
                         &f,
                         &ast_util::float_ty_to_string(t)))
            }
            ast::LitFloatUnsuffixed(ref f) => word(&mut self.s, &f[..]),
            ast::LitBool(val) => {
                if val { word(&mut self.s, "true") } else { word(&mut self.s, "false") }
            }
            ast::LitBinary(ref v) => {
                let mut escaped: String = String::new();
                for &ch in &**v {
                    escaped.extend(ascii::escape_default(ch as u8)
                                         .map(|c| c as char));
                }
                word(&mut self.s, &format!("b\"{}\"", escaped))
            }
        }
    }

    pub fn next_lit(&mut self, pos: BytePos) -> Option<comments::Literal> {
        match self.literals {
            Some(ref lits) => {
                while self.cur_cmnt_and_lit.cur_lit < lits.len() {
                    let ltrl = (*lits)[self.cur_cmnt_and_lit.cur_lit].clone();
                    if ltrl.pos > pos { return None; }
                    self.cur_cmnt_and_lit.cur_lit += 1;
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
                        self.cur_cmnt_and_lit.cur_cmnt += 1;
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
                assert_eq!(cmnt.lines.len(), 1);
                try!(zerobreak(&mut self.s));
                try!(word(&mut self.s, &cmnt.lines[0]));
                zerobreak(&mut self.s)
            }
            comments::Isolated => {
                try!(self.hardbreak_if_not_bol());
                for line in &cmnt.lines {
                    // Don't print empty lines because they will end up as trailing
                    // whitespace
                    if !line.is_empty() {
                        try!(word(&mut self.s, &line[..]));
                    }
                    try!(hardbreak(&mut self.s));
                }
                Ok(())
            }
            comments::Trailing => {
                try!(word(&mut self.s, " "));
                if cmnt.lines.len() == 1 {
                    try!(word(&mut self.s, &cmnt.lines[0]));
                    hardbreak(&mut self.s)
                } else {
                    try!(self.ibox(0));
                    for line in &cmnt.lines {
                        if !line.is_empty() {
                            try!(word(&mut self.s, &line[..]));
                        }
                        try!(hardbreak(&mut self.s));
                    }
                    self.end()
                }
            }
            comments::BlankLine => {
                // We need to do at least one, possibly two hardbreaks.
                let is_semi = match self.s.last_token() {
                    pp::Token::String(s, _) => ";" == s,
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
                         delim=repeat("#", n),
                         string=st))
            }
        };
        word(&mut self.s, &st[..])
    }

    pub fn next_comment(&mut self) -> Option<comments::Comment> {
        match self.comments {
            Some(ref cmnts) => {
                if self.cur_cmnt_and_lit.cur_cmnt < cmnts.len() {
                    Some(cmnts[self.cur_cmnt_and_lit.cur_cmnt].clone())
                } else {
                    None
                }
            }
            _ => None
        }
    }

    pub fn print_opt_unsafety(&mut self,
                            opt_unsafety: Option<ast::Unsafety>) -> IoResult<()> {
        match opt_unsafety {
            Some(unsafety) => self.print_unsafety(unsafety),
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
                self.word_nbsp(&abi.to_string())
            }
            None => Ok(())
        }
    }

    pub fn print_extern_opt_abi(&mut self,
                                opt_abi: Option<abi::Abi>) -> IoResult<()> {
        match opt_abi {
            Some(abi) => {
                try!(self.word_nbsp("extern"));
                self.word_nbsp(&abi.to_string())
            }
            None => Ok(())
        }
    }

    pub fn print_fn_header_info(&mut self,
                                opt_unsafety: Option<ast::Unsafety>,
                                abi: abi::Abi,
                                vis: ast::Visibility) -> IoResult<()> {
        try!(word(&mut self.s, &visibility_qualified(vis, "")));
        try!(self.print_opt_unsafety(opt_unsafety));

        if abi != abi::Rust {
            try!(self.word_nbsp("extern"));
            try!(self.word_nbsp(&abi.to_string()));
        }

        word(&mut self.s, "fn")
    }

    pub fn print_unsafety(&mut self, s: ast::Unsafety) -> IoResult<()> {
        match s {
            ast::Unsafety::Normal => Ok(()),
            ast::Unsafety::Unsafe => self.word_nbsp("unsafe"),
        }
    }
}

fn repeat(s: &str, n: usize) -> String { iter::repeat(s).take(n).collect() }

#[cfg(test)]
mod test {
    use super::*;

    use ast;
    use ast_util;
    use codemap;
    use parse::token;

    #[test]
    fn test_fun_to_string() {
        let abba_ident = token::str_to_ident("abba");

        let decl = ast::FnDecl {
            inputs: Vec::new(),
            output: ast::DefaultReturn(codemap::DUMMY_SP),
            variadic: false
        };
        let generics = ast_util::empty_generics();
        assert_eq!(fun_to_string(&decl, ast::Unsafety::Normal, abba_ident,
                               None, &generics),
                   "fn abba()");
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
        assert_eq!(varstr, "pub principal_skinner");
    }

    #[test]
    fn test_signed_int_to_string() {
        let pos_int = ast::LitInt(42, ast::SignedIntLit(ast::TyI32, ast::Plus));
        let neg_int = ast::LitInt((-42) as u64, ast::SignedIntLit(ast::TyI32, ast::Minus));
        assert_eq!(format!("-{}", lit_to_string(&codemap::dummy_spanned(pos_int))),
                   lit_to_string(&codemap::dummy_spanned(neg_int)));
    }
}
