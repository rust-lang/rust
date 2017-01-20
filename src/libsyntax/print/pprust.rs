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

use abi::{self, Abi};
use ast::{self, BlockCheckMode, PatKind};
use ast::{SelfKind, RegionTyParamBound, TraitTyParamBound, TraitBoundModifier};
use ast::Attribute;
use util::parser::AssocOp;
use attr;
use codemap::{self, CodeMap};
use syntax_pos::{self, BytePos};
use parse::token::{self, BinOpToken, Token};
use parse::lexer::comments;
use parse::{self, ParseSess};
use print::pp::{self, break_offset, word, space, zerobreak, hardbreak};
use print::pp::{Breaks, eof};
use print::pp::Breaks::{Consistent, Inconsistent};
use ptr::P;
use std_inject;
use symbol::{Symbol, keywords};
use tokenstream::{self, TokenTree};

use rustc_i128::i128;

use std::ascii;
use std::io::{self, Write, Read};
use std::iter;

pub enum AnnNode<'a> {
    NodeIdent(&'a ast::Ident),
    NodeName(&'a ast::Name),
    NodeBlock(&'a ast::Block),
    NodeItem(&'a ast::Item),
    NodeSubItem(ast::NodeId),
    NodeExpr(&'a ast::Expr),
    NodePat(&'a ast::Pat),
}

pub trait PpAnn {
    fn pre(&self, _state: &mut State, _node: AnnNode) -> io::Result<()> { Ok(()) }
    fn post(&self, _state: &mut State, _node: AnnNode) -> io::Result<()> { Ok(()) }
}

#[derive(Copy, Clone)]
pub struct NoAnn;

impl PpAnn for NoAnn {}

#[derive(Copy, Clone)]
pub struct CurrentCommentAndLiteral {
    pub cur_cmnt: usize,
    pub cur_lit: usize,
}

pub struct State<'a> {
    pub s: pp::Printer<'a>,
    cm: Option<&'a CodeMap>,
    comments: Option<Vec<comments::Comment> >,
    literals: Option<Vec<comments::Literal> >,
    cur_cmnt_and_lit: CurrentCommentAndLiteral,
    boxes: Vec<pp::Breaks>,
    ann: &'a (PpAnn+'a),
}

pub fn rust_printer<'a>(writer: Box<Write+'a>) -> State<'a> {
    static NO_ANN: NoAnn = NoAnn;
    rust_printer_annotated(writer, &NO_ANN)
}

pub fn rust_printer_annotated<'a>(writer: Box<Write+'a>,
                                  ann: &'a PpAnn) -> State<'a> {
    State {
        s: pp::mk_printer(writer, DEFAULT_COLUMNS),
        cm: None,
        comments: None,
        literals: None,
        cur_cmnt_and_lit: CurrentCommentAndLiteral {
            cur_cmnt: 0,
            cur_lit: 0
        },
        boxes: Vec::new(),
        ann: ann,
    }
}

pub const INDENT_UNIT: usize = 4;

pub const DEFAULT_COLUMNS: usize = 78;

/// Requires you to pass an input filename and reader so that
/// it can scan the input text for comments and literals to
/// copy forward.
pub fn print_crate<'a>(cm: &'a CodeMap,
                       sess: &ParseSess,
                       krate: &ast::Crate,
                       filename: String,
                       input: &mut Read,
                       out: Box<Write+'a>,
                       ann: &'a PpAnn,
                       is_expanded: bool) -> io::Result<()> {
    let mut s = State::new_from_input(cm, sess, filename, input, out, ann, is_expanded);

    if is_expanded && !std_inject::injected_crate_name(krate).is_none() {
        // We need to print `#![no_std]` (and its feature gate) so that
        // compiling pretty-printed source won't inject libstd again.
        // However we don't want these attributes in the AST because
        // of the feature gate, so we fake them up here.

        // #![feature(prelude_import)]
        let prelude_import_meta = attr::mk_list_word_item(Symbol::intern("prelude_import"));
        let list = attr::mk_list_item(Symbol::intern("feature"), vec![prelude_import_meta]);
        let fake_attr = attr::mk_attr_inner(attr::mk_attr_id(), list);
        s.print_attribute(&fake_attr)?;

        // #![no_std]
        let no_std_meta = attr::mk_word_item(Symbol::intern("no_std"));
        let fake_attr = attr::mk_attr_inner(attr::mk_attr_id(), no_std_meta);
        s.print_attribute(&fake_attr)?;
    }

    s.print_mod(&krate.module, &krate.attrs)?;
    s.print_remaining_comments()?;
    eof(&mut s.s)
}

impl<'a> State<'a> {
    pub fn new_from_input(cm: &'a CodeMap,
                          sess: &ParseSess,
                          filename: String,
                          input: &mut Read,
                          out: Box<Write+'a>,
                          ann: &'a PpAnn,
                          is_expanded: bool) -> State<'a> {
        let (cmnts, lits) = comments::gather_comments_and_literals(sess, filename, input);

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
               out: Box<Write+'a>,
               ann: &'a PpAnn,
               comments: Option<Vec<comments::Comment>>,
               literals: Option<Vec<comments::Literal>>) -> State<'a> {
        State {
            s: pp::mk_printer(out, DEFAULT_COLUMNS),
            cm: Some(cm),
            comments: comments,
            literals: literals,
            cur_cmnt_and_lit: CurrentCommentAndLiteral {
                cur_cmnt: 0,
                cur_lit: 0
            },
            boxes: Vec::new(),
            ann: ann,
        }
    }
}

pub fn to_string<F>(f: F) -> String where
    F: FnOnce(&mut State) -> io::Result<()>,
{
    let mut wr = Vec::new();
    {
        let mut printer = rust_printer(Box::new(&mut wr));
        f(&mut printer).unwrap();
        eof(&mut printer.s).unwrap();
    }
    String::from_utf8(wr).unwrap()
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
        token::OpenDelim(token::NoDelim) => " ".to_string(),
        token::CloseDelim(token::NoDelim) => " ".to_string(),
        token::Pound                => "#".to_string(),
        token::Dollar               => "$".to_string(),
        token::Question             => "?".to_string(),

        /* Literals */
        token::Literal(lit, suf) => {
            let mut out = match lit {
                token::Byte(b)           => format!("b'{}'", b),
                token::Char(c)           => format!("'{}'", c),
                token::Float(c)          => c.to_string(),
                token::Integer(c)        => c.to_string(),
                token::Str_(s)           => format!("\"{}\"", s),
                token::StrRaw(s, n)      => format!("r{delim}\"{string}\"{delim}",
                                                    delim=repeat("#", n),
                                                    string=s),
                token::ByteStr(v)         => format!("b\"{}\"", v),
                token::ByteStrRaw(s, n)   => format!("br{delim}\"{string}\"{delim}",
                                                    delim=repeat("#", n),
                                                    string=s),
            };

            if let Some(s) = suf {
                out.push_str(&s.as_str())
            }

            out
        }

        /* Name components */
        token::Ident(s)             => s.to_string(),
        token::Lifetime(s)          => s.to_string(),
        token::Underscore           => "_".to_string(),

        /* Other */
        token::DocComment(s)        => s.to_string(),
        token::SubstNt(s)           => format!("${}", s),
        token::MatchNt(s, t)        => format!("${}:{}", s, t),
        token::Eof                  => "<eof>".to_string(),
        token::Whitespace           => " ".to_string(),
        token::Comment              => "/* */".to_string(),
        token::Shebang(s)           => format!("/* shebang: {}*/", s),

        token::Interpolated(ref nt) => match **nt {
            token::NtExpr(ref e)        => expr_to_string(&e),
            token::NtMeta(ref e)        => meta_item_to_string(&e),
            token::NtTy(ref e)          => ty_to_string(&e),
            token::NtPath(ref e)        => path_to_string(&e),
            token::NtItem(ref e)        => item_to_string(&e),
            token::NtBlock(ref e)       => block_to_string(&e),
            token::NtStmt(ref e)        => stmt_to_string(&e),
            token::NtPat(ref e)         => pat_to_string(&e),
            token::NtIdent(ref e)       => ident_to_string(e.node),
            token::NtTT(ref e)          => tt_to_string(&e),
            token::NtArm(ref e)         => arm_to_string(&e),
            token::NtImplItem(ref e)    => impl_item_to_string(&e),
            token::NtTraitItem(ref e)   => trait_item_to_string(&e),
            token::NtGenerics(ref e)    => generics_to_string(&e),
            token::NtWhereClause(ref e) => where_clause_to_string(&e),
            token::NtArg(ref e)         => arg_to_string(&e),
        }
    }
}

pub fn ty_to_string(ty: &ast::Ty) -> String {
    to_string(|s| s.print_type(ty))
}

pub fn bounds_to_string(bounds: &[ast::TyParamBound]) -> String {
    to_string(|s| s.print_bounds("", bounds))
}

pub fn pat_to_string(pat: &ast::Pat) -> String {
    to_string(|s| s.print_pat(pat))
}

pub fn arm_to_string(arm: &ast::Arm) -> String {
    to_string(|s| s.print_arm(arm))
}

pub fn expr_to_string(e: &ast::Expr) -> String {
    to_string(|s| s.print_expr(e))
}

pub fn lifetime_to_string(e: &ast::Lifetime) -> String {
    to_string(|s| s.print_lifetime(e))
}

pub fn tt_to_string(tt: &tokenstream::TokenTree) -> String {
    to_string(|s| s.print_tt(tt))
}

pub fn tts_to_string(tts: &[tokenstream::TokenTree]) -> String {
    to_string(|s| s.print_tts(tts))
}

pub fn stmt_to_string(stmt: &ast::Stmt) -> String {
    to_string(|s| s.print_stmt(stmt))
}

pub fn attr_to_string(attr: &ast::Attribute) -> String {
    to_string(|s| s.print_attribute(attr))
}

pub fn item_to_string(i: &ast::Item) -> String {
    to_string(|s| s.print_item(i))
}

pub fn impl_item_to_string(i: &ast::ImplItem) -> String {
    to_string(|s| s.print_impl_item(i))
}

pub fn trait_item_to_string(i: &ast::TraitItem) -> String {
    to_string(|s| s.print_trait_item(i))
}

pub fn generics_to_string(generics: &ast::Generics) -> String {
    to_string(|s| s.print_generics(generics))
}

pub fn where_clause_to_string(i: &ast::WhereClause) -> String {
    to_string(|s| s.print_where_clause(i))
}

pub fn fn_block_to_string(p: &ast::FnDecl) -> String {
    to_string(|s| s.print_fn_block_args(p))
}

pub fn path_to_string(p: &ast::Path) -> String {
    to_string(|s| s.print_path(p, false, 0, false))
}

pub fn ident_to_string(id: ast::Ident) -> String {
    to_string(|s| s.print_ident(id))
}

pub fn fun_to_string(decl: &ast::FnDecl,
                     unsafety: ast::Unsafety,
                     constness: ast::Constness,
                     name: ast::Ident,
                     generics: &ast::Generics)
                     -> String {
    to_string(|s| {
        s.head("")?;
        s.print_fn(decl, unsafety, constness, Abi::Rust, Some(name),
                   generics, &ast::Visibility::Inherited)?;
        s.end()?; // Close the head box
        s.end() // Close the outer box
    })
}

pub fn block_to_string(blk: &ast::Block) -> String {
    to_string(|s| {
        // containing cbox, will be closed by print-block at }
        s.cbox(INDENT_UNIT)?;
        // head-ibox, will be closed by print-block after {
        s.ibox(0)?;
        s.print_block(blk)
    })
}

pub fn meta_list_item_to_string(li: &ast::NestedMetaItem) -> String {
    to_string(|s| s.print_meta_list_item(li))
}

pub fn meta_item_to_string(mi: &ast::MetaItem) -> String {
    to_string(|s| s.print_meta_item(mi))
}

pub fn attribute_to_string(attr: &ast::Attribute) -> String {
    to_string(|s| s.print_attribute(attr))
}

pub fn lit_to_string(l: &ast::Lit) -> String {
    to_string(|s| s.print_literal(l))
}

pub fn variant_to_string(var: &ast::Variant) -> String {
    to_string(|s| s.print_variant(var))
}

pub fn arg_to_string(arg: &ast::Arg) -> String {
    to_string(|s| s.print_arg(arg, false))
}

pub fn mac_to_string(arg: &ast::Mac) -> String {
    to_string(|s| s.print_mac(arg, ::parse::token::Paren))
}

pub fn visibility_qualified(vis: &ast::Visibility, s: &str) -> String {
    match *vis {
        ast::Visibility::Public => format!("pub {}", s),
        ast::Visibility::Crate(_) => format!("pub(crate) {}", s),
        ast::Visibility::Restricted { ref path, .. } =>
            format!("pub({}) {}", to_string(|s| s.print_path(path, false, 0, true)), s),
        ast::Visibility::Inherited => s.to_string()
    }
}

fn needs_parentheses(expr: &ast::Expr) -> bool {
    match expr.node {
        ast::ExprKind::Assign(..) | ast::ExprKind::Binary(..) |
        ast::ExprKind::Closure(..) |
        ast::ExprKind::AssignOp(..) | ast::ExprKind::Cast(..) |
        ast::ExprKind::InPlace(..) | ast::ExprKind::Type(..) => true,
        _ => false,
    }
}

pub trait PrintState<'a> {
    fn writer(&mut self) -> &mut pp::Printer<'a>;
    fn boxes(&mut self) -> &mut Vec<pp::Breaks>;
    fn comments(&mut self) -> &mut Option<Vec<comments::Comment>>;
    fn cur_cmnt_and_lit(&mut self) -> &mut CurrentCommentAndLiteral;
    fn literals(&self) -> &Option<Vec<comments::Literal>>;

    fn word_space(&mut self, w: &str) -> io::Result<()> {
        word(self.writer(), w)?;
        space(self.writer())
    }

    fn popen(&mut self) -> io::Result<()> { word(self.writer(), "(") }

    fn pclose(&mut self) -> io::Result<()> { word(self.writer(), ")") }

    fn is_begin(&mut self) -> bool {
        match self.writer().last_token() {
            pp::Token::Begin(_) => true,
            _ => false,
        }
    }

    fn is_end(&mut self) -> bool {
        match self.writer().last_token() {
            pp::Token::End => true,
            _ => false,
        }
    }

    // is this the beginning of a line?
    fn is_bol(&mut self) -> bool {
        self.writer().last_token().is_eof() || self.writer().last_token().is_hardbreak_tok()
    }

    fn hardbreak_if_not_bol(&mut self) -> io::Result<()> {
        if !self.is_bol() {
            hardbreak(self.writer())?
        }
        Ok(())
    }

    // "raw box"
    fn rbox(&mut self, u: usize, b: pp::Breaks) -> io::Result<()> {
        self.boxes().push(b);
        pp::rbox(self.writer(), u, b)
    }

    fn ibox(&mut self, u: usize) -> io::Result<()> {
        self.boxes().push(pp::Breaks::Inconsistent);
        pp::ibox(self.writer(), u)
    }

    fn end(&mut self) -> io::Result<()> {
        self.boxes().pop().unwrap();
        pp::end(self.writer())
    }

    fn commasep<T, F>(&mut self, b: Breaks, elts: &[T], mut op: F) -> io::Result<()>
        where F: FnMut(&mut Self, &T) -> io::Result<()>,
    {
        self.rbox(0, b)?;
        let mut first = true;
        for elt in elts {
            if first { first = false; } else { self.word_space(",")?; }
            op(self, elt)?;
        }
        self.end()
    }

    fn next_lit(&mut self, pos: BytePos) -> Option<comments::Literal> {
        let mut cur_lit = self.cur_cmnt_and_lit().cur_lit;

        let mut result = None;

        if let &Some(ref lits) = self.literals()
        {
            while cur_lit < lits.len() {
                let ltrl = (*lits)[cur_lit].clone();
                if ltrl.pos > pos { break; }
                cur_lit += 1;
                if ltrl.pos == pos {
                    result = Some(ltrl);
                    break;
                }
            }
        }

        self.cur_cmnt_and_lit().cur_lit = cur_lit;
        result
    }

    fn maybe_print_comment(&mut self, pos: BytePos) -> io::Result<()> {
        while let Some(ref cmnt) = self.next_comment() {
            if cmnt.pos < pos {
                self.print_comment(cmnt)?;
                self.cur_cmnt_and_lit().cur_cmnt += 1;
            } else {
                break
            }
        }
        Ok(())
    }

    fn print_comment(&mut self,
                     cmnt: &comments::Comment) -> io::Result<()> {
        match cmnt.style {
            comments::Mixed => {
                assert_eq!(cmnt.lines.len(), 1);
                zerobreak(self.writer())?;
                word(self.writer(), &cmnt.lines[0])?;
                zerobreak(self.writer())
            }
            comments::Isolated => {
                self.hardbreak_if_not_bol()?;
                for line in &cmnt.lines {
                    // Don't print empty lines because they will end up as trailing
                    // whitespace
                    if !line.is_empty() {
                        word(self.writer(), &line[..])?;
                    }
                    hardbreak(self.writer())?;
                }
                Ok(())
            }
            comments::Trailing => {
                if !self.is_bol() {
                    word(self.writer(), " ")?;
                }
                if cmnt.lines.len() == 1 {
                    word(self.writer(), &cmnt.lines[0])?;
                    hardbreak(self.writer())
                } else {
                    self.ibox(0)?;
                    for line in &cmnt.lines {
                        if !line.is_empty() {
                            word(self.writer(), &line[..])?;
                        }
                        hardbreak(self.writer())?;
                    }
                    self.end()
                }
            }
            comments::BlankLine => {
                // We need to do at least one, possibly two hardbreaks.
                let is_semi = match self.writer().last_token() {
                    pp::Token::String(s, _) => ";" == s,
                    _ => false
                };
                if is_semi || self.is_begin() || self.is_end() {
                    hardbreak(self.writer())?;
                }
                hardbreak(self.writer())
            }
        }
    }

    fn next_comment(&mut self) -> Option<comments::Comment> {
        let cur_cmnt = self.cur_cmnt_and_lit().cur_cmnt;
        match *self.comments() {
            Some(ref cmnts) => {
                if cur_cmnt < cmnts.len() {
                    Some(cmnts[cur_cmnt].clone())
                } else {
                    None
                }
            }
            _ => None
        }
    }

    fn print_literal(&mut self, lit: &ast::Lit) -> io::Result<()> {
        self.maybe_print_comment(lit.span.lo)?;
        match self.next_lit(lit.span.lo) {
            Some(ref ltrl) => {
                return word(self.writer(), &(*ltrl).lit);
            }
            _ => ()
        }
        match lit.node {
            ast::LitKind::Str(st, style) => self.print_string(&st.as_str(), style),
            ast::LitKind::Byte(byte) => {
                let mut res = String::from("b'");
                res.extend(ascii::escape_default(byte).map(|c| c as char));
                res.push('\'');
                word(self.writer(), &res[..])
            }
            ast::LitKind::Char(ch) => {
                let mut res = String::from("'");
                res.extend(ch.escape_default());
                res.push('\'');
                word(self.writer(), &res[..])
            }
            ast::LitKind::Int(i, t) => {
                match t {
                    ast::LitIntType::Signed(st) => {
                        word(self.writer(), &st.val_to_string(i as i128))
                    }
                    ast::LitIntType::Unsigned(ut) => {
                        word(self.writer(), &ut.val_to_string(i))
                    }
                    ast::LitIntType::Unsuffixed => {
                        word(self.writer(), &format!("{}", i))
                    }
                }
            }
            ast::LitKind::Float(ref f, t) => {
                word(self.writer(),
                     &format!(
                         "{}{}",
                         &f,
                         t.ty_to_string()))
            }
            ast::LitKind::FloatUnsuffixed(ref f) => word(self.writer(), &f.as_str()),
            ast::LitKind::Bool(val) => {
                if val { word(self.writer(), "true") } else { word(self.writer(), "false") }
            }
            ast::LitKind::ByteStr(ref v) => {
                let mut escaped: String = String::new();
                for &ch in v.iter() {
                    escaped.extend(ascii::escape_default(ch)
                                         .map(|c| c as char));
                }
                word(self.writer(), &format!("b\"{}\"", escaped))
            }
        }
    }

    fn print_string(&mut self, st: &str,
                    style: ast::StrStyle) -> io::Result<()> {
        let st = match style {
            ast::StrStyle::Cooked => {
                (format!("\"{}\"", st.escape_default()))
            }
            ast::StrStyle::Raw(n) => {
                (format!("r{delim}\"{string}\"{delim}",
                         delim=repeat("#", n),
                         string=st))
            }
        };
        word(self.writer(), &st[..])
    }

    fn print_inner_attributes(&mut self,
                              attrs: &[ast::Attribute]) -> io::Result<()> {
        self.print_either_attributes(attrs, ast::AttrStyle::Inner, false, true)
    }

    fn print_inner_attributes_no_trailing_hardbreak(&mut self,
                                                   attrs: &[ast::Attribute])
                                                   -> io::Result<()> {
        self.print_either_attributes(attrs, ast::AttrStyle::Inner, false, false)
    }

    fn print_outer_attributes(&mut self,
                              attrs: &[ast::Attribute]) -> io::Result<()> {
        self.print_either_attributes(attrs, ast::AttrStyle::Outer, false, true)
    }

    fn print_inner_attributes_inline(&mut self,
                                     attrs: &[ast::Attribute]) -> io::Result<()> {
        self.print_either_attributes(attrs, ast::AttrStyle::Inner, true, true)
    }

    fn print_outer_attributes_inline(&mut self,
                                     attrs: &[ast::Attribute]) -> io::Result<()> {
        self.print_either_attributes(attrs, ast::AttrStyle::Outer, true, true)
    }

    fn print_either_attributes(&mut self,
                              attrs: &[ast::Attribute],
                              kind: ast::AttrStyle,
                              is_inline: bool,
                              trailing_hardbreak: bool) -> io::Result<()> {
        let mut count = 0;
        for attr in attrs {
            if attr.style == kind {
                self.print_attribute_inline(attr, is_inline)?;
                if is_inline {
                    self.nbsp()?;
                }
                count += 1;
            }
        }
        if count > 0 && trailing_hardbreak && !is_inline {
            self.hardbreak_if_not_bol()?;
        }
        Ok(())
    }

    fn print_attribute(&mut self, attr: &ast::Attribute) -> io::Result<()> {
        self.print_attribute_inline(attr, false)
    }

    fn print_attribute_inline(&mut self, attr: &ast::Attribute,
                              is_inline: bool) -> io::Result<()> {
        if !is_inline {
            self.hardbreak_if_not_bol()?;
        }
        self.maybe_print_comment(attr.span.lo)?;
        if attr.is_sugared_doc {
            word(self.writer(), &attr.value_str().unwrap().as_str())?;
            hardbreak(self.writer())
        } else {
            match attr.style {
                ast::AttrStyle::Inner => word(self.writer(), "#![")?,
                ast::AttrStyle::Outer => word(self.writer(), "#[")?,
            }
            self.print_meta_item(&attr.meta())?;
            word(self.writer(), "]")
        }
    }

    fn print_meta_list_item(&mut self, item: &ast::NestedMetaItem) -> io::Result<()> {
        match item.node {
            ast::NestedMetaItemKind::MetaItem(ref mi) => {
                self.print_meta_item(mi)
            },
            ast::NestedMetaItemKind::Literal(ref lit) => {
                self.print_literal(lit)
            }
        }
    }

    fn print_meta_item(&mut self, item: &ast::MetaItem) -> io::Result<()> {
        self.ibox(INDENT_UNIT)?;
        match item.node {
            ast::MetaItemKind::Word => {
                word(self.writer(), &item.name.as_str())?;
            }
            ast::MetaItemKind::NameValue(ref value) => {
                self.word_space(&item.name.as_str())?;
                self.word_space("=")?;
                self.print_literal(value)?;
            }
            ast::MetaItemKind::List(ref items) => {
                word(self.writer(), &item.name.as_str())?;
                self.popen()?;
                self.commasep(Consistent,
                              &items[..],
                              |s, i| s.print_meta_list_item(&i))?;
                self.pclose()?;
            }
        }
        self.end()
    }

    fn space_if_not_bol(&mut self) -> io::Result<()> {
        if !self.is_bol() { space(self.writer())?; }
        Ok(())
    }

    fn nbsp(&mut self) -> io::Result<()> { word(self.writer(), " ") }
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

    fn cur_cmnt_and_lit(&mut self) -> &mut CurrentCommentAndLiteral {
        &mut self.cur_cmnt_and_lit
    }

    fn literals(&self) -> &Option<Vec<comments::Literal>> {
        &self.literals
    }
}

impl<'a> State<'a> {
    pub fn cbox(&mut self, u: usize) -> io::Result<()> {
        self.boxes.push(pp::Breaks::Consistent);
        pp::cbox(&mut self.s, u)
    }

    pub fn word_nbsp(&mut self, w: &str) -> io::Result<()> {
        word(&mut self.s, w)?;
        self.nbsp()
    }

    pub fn head(&mut self, w: &str) -> io::Result<()> {
        // outer-box is consistent
        self.cbox(INDENT_UNIT)?;
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

    pub fn bclose_(&mut self, span: syntax_pos::Span,
                   indented: usize) -> io::Result<()> {
        self.bclose_maybe_open(span, indented, true)
    }
    pub fn bclose_maybe_open(&mut self, span: syntax_pos::Span,
                             indented: usize, close_box: bool) -> io::Result<()> {
        self.maybe_print_comment(span.hi)?;
        self.break_offset_if_not_bol(1, -(indented as isize))?;
        word(&mut self.s, "}")?;
        if close_box {
            self.end()?; // close the outer-box
        }
        Ok(())
    }
    pub fn bclose(&mut self, span: syntax_pos::Span) -> io::Result<()> {
        self.bclose_(span, INDENT_UNIT)
    }

    pub fn in_cbox(&self) -> bool {
        match self.boxes.last() {
            Some(&last_box) => last_box == pp::Breaks::Consistent,
            None => false
        }
    }

    pub fn break_offset_if_not_bol(&mut self, n: usize,
                                   off: isize) -> io::Result<()> {
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
                                  mut get_span: G) -> io::Result<()> where
        F: FnMut(&mut State, &T) -> io::Result<()>,
        G: FnMut(&T) -> syntax_pos::Span,
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
                self.maybe_print_trailing_comment(get_span(elt),
                                                  Some(get_span(&elts[i]).hi))?;
                self.space_if_not_bol()?;
            }
        }
        self.end()
    }

    pub fn commasep_exprs(&mut self, b: Breaks,
                          exprs: &[P<ast::Expr>]) -> io::Result<()> {
        self.commasep_cmnt(b, exprs, |s, e| s.print_expr(&e), |e| e.span)
    }

    pub fn print_mod(&mut self, _mod: &ast::Mod,
                     attrs: &[ast::Attribute]) -> io::Result<()> {
        self.print_inner_attributes(attrs)?;
        for item in &_mod.items {
            self.print_item(&item)?;
        }
        Ok(())
    }

    pub fn print_foreign_mod(&mut self, nmod: &ast::ForeignMod,
                             attrs: &[ast::Attribute]) -> io::Result<()> {
        self.print_inner_attributes(attrs)?;
        for item in &nmod.items {
            self.print_foreign_item(item)?;
        }
        Ok(())
    }

    pub fn print_opt_lifetime(&mut self,
                              lifetime: &Option<ast::Lifetime>) -> io::Result<()> {
        if let Some(l) = *lifetime {
            self.print_lifetime(&l)?;
            self.nbsp()?;
        }
        Ok(())
    }

    pub fn print_type(&mut self, ty: &ast::Ty) -> io::Result<()> {
        self.maybe_print_comment(ty.span.lo)?;
        self.ibox(0)?;
        match ty.node {
            ast::TyKind::Slice(ref ty) => {
                word(&mut self.s, "[")?;
                self.print_type(&ty)?;
                word(&mut self.s, "]")?;
            }
            ast::TyKind::Ptr(ref mt) => {
                word(&mut self.s, "*")?;
                match mt.mutbl {
                    ast::Mutability::Mutable => self.word_nbsp("mut")?,
                    ast::Mutability::Immutable => self.word_nbsp("const")?,
                }
                self.print_type(&mt.ty)?;
            }
            ast::TyKind::Rptr(ref lifetime, ref mt) => {
                word(&mut self.s, "&")?;
                self.print_opt_lifetime(lifetime)?;
                self.print_mt(mt)?;
            }
            ast::TyKind::Never => {
                word(&mut self.s, "!")?;
            },
            ast::TyKind::Tup(ref elts) => {
                self.popen()?;
                self.commasep(Inconsistent, &elts[..],
                              |s, ty| s.print_type(&ty))?;
                if elts.len() == 1 {
                    word(&mut self.s, ",")?;
                }
                self.pclose()?;
            }
            ast::TyKind::Paren(ref typ) => {
                self.popen()?;
                self.print_type(&typ)?;
                self.pclose()?;
            }
            ast::TyKind::BareFn(ref f) => {
                let generics = ast::Generics {
                    lifetimes: f.lifetimes.clone(),
                    ty_params: Vec::new(),
                    where_clause: ast::WhereClause {
                        id: ast::DUMMY_NODE_ID,
                        predicates: Vec::new(),
                    },
                    span: syntax_pos::DUMMY_SP,
                };
                self.print_ty_fn(f.abi,
                                 f.unsafety,
                                 &f.decl,
                                 None,
                                 &generics)?;
            }
            ast::TyKind::Path(None, ref path) => {
                self.print_path(path, false, 0, false)?;
            }
            ast::TyKind::Path(Some(ref qself), ref path) => {
                self.print_qpath(path, qself, false)?
            }
            ast::TyKind::TraitObject(ref bounds) => {
                self.print_bounds("", &bounds[..])?;
            }
            ast::TyKind::ImplTrait(ref bounds) => {
                self.print_bounds("impl ", &bounds[..])?;
            }
            ast::TyKind::Array(ref ty, ref v) => {
                word(&mut self.s, "[")?;
                self.print_type(&ty)?;
                word(&mut self.s, "; ")?;
                self.print_expr(&v)?;
                word(&mut self.s, "]")?;
            }
            ast::TyKind::Typeof(ref e) => {
                word(&mut self.s, "typeof(")?;
                self.print_expr(&e)?;
                word(&mut self.s, ")")?;
            }
            ast::TyKind::Infer => {
                word(&mut self.s, "_")?;
            }
            ast::TyKind::ImplicitSelf => {
                word(&mut self.s, "Self")?;
            }
            ast::TyKind::Mac(ref m) => {
                self.print_mac(m, token::Paren)?;
            }
        }
        self.end()
    }

    pub fn print_foreign_item(&mut self,
                              item: &ast::ForeignItem) -> io::Result<()> {
        self.hardbreak_if_not_bol()?;
        self.maybe_print_comment(item.span.lo)?;
        self.print_outer_attributes(&item.attrs)?;
        match item.node {
            ast::ForeignItemKind::Fn(ref decl, ref generics) => {
                self.head("")?;
                self.print_fn(decl, ast::Unsafety::Normal,
                              ast::Constness::NotConst,
                              Abi::Rust, Some(item.ident),
                              generics, &item.vis)?;
                self.end()?; // end head-ibox
                word(&mut self.s, ";")?;
                self.end() // end the outer fn box
            }
            ast::ForeignItemKind::Static(ref t, m) => {
                self.head(&visibility_qualified(&item.vis, "static"))?;
                if m {
                    self.word_space("mut")?;
                }
                self.print_ident(item.ident)?;
                self.word_space(":")?;
                self.print_type(&t)?;
                word(&mut self.s, ";")?;
                self.end()?; // end the head-ibox
                self.end() // end the outer cbox
            }
        }
    }

    fn print_associated_const(&mut self,
                              ident: ast::Ident,
                              ty: &ast::Ty,
                              default: Option<&ast::Expr>,
                              vis: &ast::Visibility)
                              -> io::Result<()>
    {
        word(&mut self.s, &visibility_qualified(vis, ""))?;
        self.word_space("const")?;
        self.print_ident(ident)?;
        self.word_space(":")?;
        self.print_type(ty)?;
        if let Some(expr) = default {
            space(&mut self.s)?;
            self.word_space("=")?;
            self.print_expr(expr)?;
        }
        word(&mut self.s, ";")
    }

    fn print_associated_type(&mut self,
                             ident: ast::Ident,
                             bounds: Option<&ast::TyParamBounds>,
                             ty: Option<&ast::Ty>)
                             -> io::Result<()> {
        self.word_space("type")?;
        self.print_ident(ident)?;
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
    pub fn print_item(&mut self, item: &ast::Item) -> io::Result<()> {
        self.hardbreak_if_not_bol()?;
        self.maybe_print_comment(item.span.lo)?;
        self.print_outer_attributes(&item.attrs)?;
        self.ann.pre(self, NodeItem(item))?;
        match item.node {
            ast::ItemKind::ExternCrate(ref optional_path) => {
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
                self.print_ident(item.ident)?;
                word(&mut self.s, ";")?;
                self.end()?; // end inner head-block
                self.end()?; // end outer head-block
            }
            ast::ItemKind::Use(ref vp) => {
                self.head(&visibility_qualified(&item.vis, "use"))?;
                self.print_view_path(&vp)?;
                word(&mut self.s, ";")?;
                self.end()?; // end inner head-block
                self.end()?; // end outer head-block
            }
            ast::ItemKind::Static(ref ty, m, ref expr) => {
                self.head(&visibility_qualified(&item.vis, "static"))?;
                if m == ast::Mutability::Mutable {
                    self.word_space("mut")?;
                }
                self.print_ident(item.ident)?;
                self.word_space(":")?;
                self.print_type(&ty)?;
                space(&mut self.s)?;
                self.end()?; // end the head-ibox

                self.word_space("=")?;
                self.print_expr(&expr)?;
                word(&mut self.s, ";")?;
                self.end()?; // end the outer cbox
            }
            ast::ItemKind::Const(ref ty, ref expr) => {
                self.head(&visibility_qualified(&item.vis, "const"))?;
                self.print_ident(item.ident)?;
                self.word_space(":")?;
                self.print_type(&ty)?;
                space(&mut self.s)?;
                self.end()?; // end the head-ibox

                self.word_space("=")?;
                self.print_expr(&expr)?;
                word(&mut self.s, ";")?;
                self.end()?; // end the outer cbox
            }
            ast::ItemKind::Fn(ref decl, unsafety, constness, abi, ref typarams, ref body) => {
                self.head("")?;
                self.print_fn(
                    decl,
                    unsafety,
                    constness.node,
                    abi,
                    Some(item.ident),
                    typarams,
                    &item.vis
                )?;
                word(&mut self.s, " ")?;
                self.print_block_with_attrs(&body, &item.attrs)?;
            }
            ast::ItemKind::Mod(ref _mod) => {
                self.head(&visibility_qualified(&item.vis, "mod"))?;
                self.print_ident(item.ident)?;
                self.nbsp()?;
                self.bopen()?;
                self.print_mod(_mod, &item.attrs)?;
                self.bclose(item.span)?;
            }
            ast::ItemKind::ForeignMod(ref nmod) => {
                self.head("extern")?;
                self.word_nbsp(&nmod.abi.to_string())?;
                self.bopen()?;
                self.print_foreign_mod(nmod, &item.attrs)?;
                self.bclose(item.span)?;
            }
            ast::ItemKind::Ty(ref ty, ref params) => {
                self.ibox(INDENT_UNIT)?;
                self.ibox(0)?;
                self.word_nbsp(&visibility_qualified(&item.vis, "type"))?;
                self.print_ident(item.ident)?;
                self.print_generics(params)?;
                self.end()?; // end the inner ibox

                self.print_where_clause(&params.where_clause)?;
                space(&mut self.s)?;
                self.word_space("=")?;
                self.print_type(&ty)?;
                word(&mut self.s, ";")?;
                self.end()?; // end the outer ibox
            }
            ast::ItemKind::Enum(ref enum_definition, ref params) => {
                self.print_enum_def(
                    enum_definition,
                    params,
                    item.ident,
                    item.span,
                    &item.vis
                )?;
            }
            ast::ItemKind::Struct(ref struct_def, ref generics) => {
                self.head(&visibility_qualified(&item.vis, "struct"))?;
                self.print_struct(&struct_def, generics, item.ident, item.span, true)?;
            }
            ast::ItemKind::Union(ref struct_def, ref generics) => {
                self.head(&visibility_qualified(&item.vis, "union"))?;
                self.print_struct(&struct_def, generics, item.ident, item.span, true)?;
            }
            ast::ItemKind::DefaultImpl(unsafety, ref trait_ref) => {
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
            ast::ItemKind::Impl(unsafety,
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
                    ast::ImplPolarity::Negative => {
                        word(&mut self.s, "!")?;
                    },
                    _ => {}
                }

                if let Some(ref t) = *opt_trait {
                    self.print_trait_ref(t)?;
                    space(&mut self.s)?;
                    self.word_space("for")?;
                }

                self.print_type(&ty)?;
                self.print_where_clause(&generics.where_clause)?;

                space(&mut self.s)?;
                self.bopen()?;
                self.print_inner_attributes(&item.attrs)?;
                for impl_item in impl_items {
                    self.print_impl_item(impl_item)?;
                }
                self.bclose(item.span)?;
            }
            ast::ItemKind::Trait(unsafety, ref generics, ref bounds, ref trait_items) => {
                self.head("")?;
                self.print_visibility(&item.vis)?;
                self.print_unsafety(unsafety)?;
                self.word_nbsp("trait")?;
                self.print_ident(item.ident)?;
                self.print_generics(generics)?;
                let mut real_bounds = Vec::with_capacity(bounds.len());
                for b in bounds.iter() {
                    if let TraitTyParamBound(ref ptr, ast::TraitBoundModifier::Maybe) = *b {
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
                    self.print_trait_item(trait_item)?;
                }
                self.bclose(item.span)?;
            }
            ast::ItemKind::Mac(codemap::Spanned { ref node, .. }) => {
                self.print_visibility(&item.vis)?;
                self.print_path(&node.path, false, 0, false)?;
                word(&mut self.s, "! ")?;
                self.print_ident(item.ident)?;
                self.cbox(INDENT_UNIT)?;
                self.popen()?;
                self.print_tts(&node.tts[..])?;
                self.pclose()?;
                word(&mut self.s, ";")?;
                self.end()?;
            }
        }
        self.ann.post(self, NodeItem(item))
    }

    fn print_trait_ref(&mut self, t: &ast::TraitRef) -> io::Result<()> {
        self.print_path(&t.path, false, 0, false)
    }

    fn print_formal_lifetime_list(&mut self, lifetimes: &[ast::LifetimeDef]) -> io::Result<()> {
        if !lifetimes.is_empty() {
            word(&mut self.s, "for<")?;
            let mut comma = false;
            for lifetime_def in lifetimes {
                if comma {
                    self.word_space(",")?
                }
                self.print_outer_attributes_inline(&lifetime_def.attrs)?;
                self.print_lifetime_bounds(&lifetime_def.lifetime, &lifetime_def.bounds)?;
                comma = true;
            }
            word(&mut self.s, ">")?;
        }
        Ok(())
    }

    fn print_poly_trait_ref(&mut self, t: &ast::PolyTraitRef) -> io::Result<()> {
        self.print_formal_lifetime_list(&t.bound_lifetimes)?;
        self.print_trait_ref(&t.trait_ref)
    }

    pub fn print_enum_def(&mut self, enum_definition: &ast::EnumDef,
                          generics: &ast::Generics, ident: ast::Ident,
                          span: syntax_pos::Span,
                          visibility: &ast::Visibility) -> io::Result<()> {
        self.head(&visibility_qualified(visibility, "enum"))?;
        self.print_ident(ident)?;
        self.print_generics(generics)?;
        self.print_where_clause(&generics.where_clause)?;
        space(&mut self.s)?;
        self.print_variants(&enum_definition.variants, span)
    }

    pub fn print_variants(&mut self,
                          variants: &[ast::Variant],
                          span: syntax_pos::Span) -> io::Result<()> {
        self.bopen()?;
        for v in variants {
            self.space_if_not_bol()?;
            self.maybe_print_comment(v.span.lo)?;
            self.print_outer_attributes(&v.node.attrs)?;
            self.ibox(INDENT_UNIT)?;
            self.print_variant(v)?;
            word(&mut self.s, ",")?;
            self.end()?;
            self.maybe_print_trailing_comment(v.span, None)?;
        }
        self.bclose(span)
    }

    pub fn print_visibility(&mut self, vis: &ast::Visibility) -> io::Result<()> {
        match *vis {
            ast::Visibility::Public => self.word_nbsp("pub"),
            ast::Visibility::Crate(_) => self.word_nbsp("pub(crate)"),
            ast::Visibility::Restricted { ref path, .. } => {
                let path = to_string(|s| s.print_path(path, false, 0, true));
                self.word_nbsp(&format!("pub({})", path))
            }
            ast::Visibility::Inherited => Ok(())
        }
    }

    pub fn print_struct(&mut self,
                        struct_def: &ast::VariantData,
                        generics: &ast::Generics,
                        ident: ast::Ident,
                        span: syntax_pos::Span,
                        print_finalizer: bool) -> io::Result<()> {
        self.print_ident(ident)?;
        self.print_generics(generics)?;
        if !struct_def.is_struct() {
            if struct_def.is_tuple() {
                self.popen()?;
                self.commasep(
                    Inconsistent, struct_def.fields(),
                    |s, field| {
                        s.maybe_print_comment(field.span.lo)?;
                        s.print_outer_attributes(&field.attrs)?;
                        s.print_visibility(&field.vis)?;
                        s.print_type(&field.ty)
                    }
                )?;
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
                self.print_ident(field.ident.unwrap())?;
                self.word_nbsp(":")?;
                self.print_type(&field.ty)?;
                word(&mut self.s, ",")?;
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
    pub fn print_tt(&mut self, tt: &tokenstream::TokenTree) -> io::Result<()> {
        match *tt {
            TokenTree::Token(_, ref tk) => {
                word(&mut self.s, &token_to_string(tk))?;
                match *tk {
                    parse::token::DocComment(..) => {
                        hardbreak(&mut self.s)
                    }
                    _ => Ok(())
                }
            }
            TokenTree::Delimited(_, ref delimed) => {
                word(&mut self.s, &token_to_string(&delimed.open_token()))?;
                space(&mut self.s)?;
                self.print_tts(&delimed.tts)?;
                space(&mut self.s)?;
                word(&mut self.s, &token_to_string(&delimed.close_token()))
            },
            TokenTree::Sequence(_, ref seq) => {
                word(&mut self.s, "$(")?;
                for tt_elt in &seq.tts {
                    self.print_tt(tt_elt)?;
                }
                word(&mut self.s, ")")?;
                if let Some(ref tk) = seq.separator {
                    word(&mut self.s, &token_to_string(tk))?;
                }
                match seq.op {
                    tokenstream::KleeneOp::ZeroOrMore => word(&mut self.s, "*"),
                    tokenstream::KleeneOp::OneOrMore => word(&mut self.s, "+"),
                }
            }
        }
    }

    pub fn print_tts(&mut self, tts: &[tokenstream::TokenTree]) -> io::Result<()> {
        self.ibox(0)?;
        for (i, tt) in tts.iter().enumerate() {
            if i != 0 {
                space(&mut self.s)?;
            }
            self.print_tt(tt)?;
        }
        self.end()
    }

    pub fn print_variant(&mut self, v: &ast::Variant) -> io::Result<()> {
        self.head("")?;
        let generics = ast::Generics::default();
        self.print_struct(&v.node.data, &generics, v.node.name, v.span, false)?;
        match v.node.disr_expr {
            Some(ref d) => {
                space(&mut self.s)?;
                self.word_space("=")?;
                self.print_expr(&d)
            }
            _ => Ok(())
        }
    }

    pub fn print_method_sig(&mut self,
                            ident: ast::Ident,
                            m: &ast::MethodSig,
                            vis: &ast::Visibility)
                            -> io::Result<()> {
        self.print_fn(&m.decl,
                      m.unsafety,
                      m.constness.node,
                      m.abi,
                      Some(ident),
                      &m.generics,
                      vis)
    }

    pub fn print_trait_item(&mut self, ti: &ast::TraitItem)
                            -> io::Result<()> {
        self.ann.pre(self, NodeSubItem(ti.id))?;
        self.hardbreak_if_not_bol()?;
        self.maybe_print_comment(ti.span.lo)?;
        self.print_outer_attributes(&ti.attrs)?;
        match ti.node {
            ast::TraitItemKind::Const(ref ty, ref default) => {
                self.print_associated_const(ti.ident, &ty,
                                            default.as_ref().map(|expr| &**expr),
                                            &ast::Visibility::Inherited)?;
            }
            ast::TraitItemKind::Method(ref sig, ref body) => {
                if body.is_some() {
                    self.head("")?;
                }
                self.print_method_sig(ti.ident, sig, &ast::Visibility::Inherited)?;
                if let Some(ref body) = *body {
                    self.nbsp()?;
                    self.print_block_with_attrs(body, &ti.attrs)?;
                } else {
                    word(&mut self.s, ";")?;
                }
            }
            ast::TraitItemKind::Type(ref bounds, ref default) => {
                self.print_associated_type(ti.ident, Some(bounds),
                                           default.as_ref().map(|ty| &**ty))?;
            }
            ast::TraitItemKind::Macro(codemap::Spanned { ref node, .. }) => {
                // code copied from ItemKind::Mac:
                self.print_path(&node.path, false, 0, false)?;
                word(&mut self.s, "! ")?;
                self.cbox(INDENT_UNIT)?;
                self.popen()?;
                self.print_tts(&node.tts[..])?;
                self.pclose()?;
                word(&mut self.s, ";")?;
                self.end()?
            }
        }
        self.ann.post(self, NodeSubItem(ti.id))
    }

    pub fn print_impl_item(&mut self, ii: &ast::ImplItem) -> io::Result<()> {
        self.ann.pre(self, NodeSubItem(ii.id))?;
        self.hardbreak_if_not_bol()?;
        self.maybe_print_comment(ii.span.lo)?;
        self.print_outer_attributes(&ii.attrs)?;
        if let ast::Defaultness::Default = ii.defaultness {
            self.word_nbsp("default")?;
        }
        match ii.node {
            ast::ImplItemKind::Const(ref ty, ref expr) => {
                self.print_associated_const(ii.ident, &ty, Some(&expr), &ii.vis)?;
            }
            ast::ImplItemKind::Method(ref sig, ref body) => {
                self.head("")?;
                self.print_method_sig(ii.ident, sig, &ii.vis)?;
                self.nbsp()?;
                self.print_block_with_attrs(body, &ii.attrs)?;
            }
            ast::ImplItemKind::Type(ref ty) => {
                self.print_associated_type(ii.ident, None, Some(ty))?;
            }
            ast::ImplItemKind::Macro(codemap::Spanned { ref node, .. }) => {
                // code copied from ItemKind::Mac:
                self.print_path(&node.path, false, 0, false)?;
                word(&mut self.s, "! ")?;
                self.cbox(INDENT_UNIT)?;
                self.popen()?;
                self.print_tts(&node.tts[..])?;
                self.pclose()?;
                word(&mut self.s, ";")?;
                self.end()?
            }
        }
        self.ann.post(self, NodeSubItem(ii.id))
    }

    pub fn print_stmt(&mut self, st: &ast::Stmt) -> io::Result<()> {
        self.maybe_print_comment(st.span.lo)?;
        match st.node {
            ast::StmtKind::Local(ref loc) => {
                self.print_outer_attributes(&loc.attrs)?;
                self.space_if_not_bol()?;
                self.ibox(INDENT_UNIT)?;
                self.word_nbsp("let")?;

                self.ibox(INDENT_UNIT)?;
                self.print_local_decl(&loc)?;
                self.end()?;
                if let Some(ref init) = loc.init {
                    self.nbsp()?;
                    self.word_space("=")?;
                    self.print_expr(&init)?;
                }
                word(&mut self.s, ";")?;
                self.end()?;
            }
            ast::StmtKind::Item(ref item) => self.print_item(&item)?,
            ast::StmtKind::Expr(ref expr) => {
                self.space_if_not_bol()?;
                self.print_expr_outer_attr_style(&expr, false)?;
                if parse::classify::expr_requires_semi_to_be_stmt(expr) {
                    word(&mut self.s, ";")?;
                }
            }
            ast::StmtKind::Semi(ref expr) => {
                self.space_if_not_bol()?;
                self.print_expr_outer_attr_style(&expr, false)?;
                word(&mut self.s, ";")?;
            }
            ast::StmtKind::Mac(ref mac) => {
                let (ref mac, style, ref attrs) = **mac;
                self.space_if_not_bol()?;
                self.print_outer_attributes(&attrs)?;
                let delim = match style {
                    ast::MacStmtStyle::Braces => token::Brace,
                    _ => token::Paren
                };
                self.print_mac(&mac, delim)?;
                if style == ast::MacStmtStyle::Semicolon {
                    word(&mut self.s, ";")?;
                }
            }
        }
        self.maybe_print_trailing_comment(st.span, None)
    }

    pub fn print_block(&mut self, blk: &ast::Block) -> io::Result<()> {
        self.print_block_with_attrs(blk, &[])
    }

    pub fn print_block_unclosed(&mut self, blk: &ast::Block) -> io::Result<()> {
        self.print_block_unclosed_indent(blk, INDENT_UNIT)
    }

    pub fn print_block_unclosed_with_attrs(&mut self, blk: &ast::Block,
                                            attrs: &[ast::Attribute])
                                           -> io::Result<()> {
        self.print_block_maybe_unclosed(blk, INDENT_UNIT, attrs, false)
    }

    pub fn print_block_unclosed_indent(&mut self, blk: &ast::Block,
                                       indented: usize) -> io::Result<()> {
        self.print_block_maybe_unclosed(blk, indented, &[], false)
    }

    pub fn print_block_with_attrs(&mut self,
                                  blk: &ast::Block,
                                  attrs: &[ast::Attribute]) -> io::Result<()> {
        self.print_block_maybe_unclosed(blk, INDENT_UNIT, attrs, true)
    }

    pub fn print_block_maybe_unclosed(&mut self,
                                      blk: &ast::Block,
                                      indented: usize,
                                      attrs: &[ast::Attribute],
                                      close_box: bool) -> io::Result<()> {
        match blk.rules {
            BlockCheckMode::Unsafe(..) => self.word_space("unsafe")?,
            BlockCheckMode::Default => ()
        }
        self.maybe_print_comment(blk.span.lo)?;
        self.ann.pre(self, NodeBlock(blk))?;
        self.bopen()?;

        self.print_inner_attributes(attrs)?;

        for (i, st) in blk.stmts.iter().enumerate() {
            match st.node {
                ast::StmtKind::Expr(ref expr) if i == blk.stmts.len() - 1 => {
                    self.maybe_print_comment(st.span.lo)?;
                    self.space_if_not_bol()?;
                    self.print_expr_outer_attr_style(&expr, false)?;
                    self.maybe_print_trailing_comment(expr.span, Some(blk.span.hi))?;
                }
                _ => self.print_stmt(st)?,
            }
        }

        self.bclose_maybe_open(blk.span, indented, close_box)?;
        self.ann.post(self, NodeBlock(blk))
    }

    fn print_else(&mut self, els: Option<&ast::Expr>) -> io::Result<()> {
        match els {
            Some(_else) => {
                match _else.node {
                    // "another else-if"
                    ast::ExprKind::If(ref i, ref then, ref e) => {
                        self.cbox(INDENT_UNIT - 1)?;
                        self.ibox(0)?;
                        word(&mut self.s, " else if ")?;
                        self.print_expr(&i)?;
                        space(&mut self.s)?;
                        self.print_block(&then)?;
                        self.print_else(e.as_ref().map(|e| &**e))
                    }
                    // "another else-if-let"
                    ast::ExprKind::IfLet(ref pat, ref expr, ref then, ref e) => {
                        self.cbox(INDENT_UNIT - 1)?;
                        self.ibox(0)?;
                        word(&mut self.s, " else if let ")?;
                        self.print_pat(&pat)?;
                        space(&mut self.s)?;
                        self.word_space("=")?;
                        self.print_expr(&expr)?;
                        space(&mut self.s)?;
                        self.print_block(&then)?;
                        self.print_else(e.as_ref().map(|e| &**e))
                    }
                    // "final else"
                    ast::ExprKind::Block(ref b) => {
                        self.cbox(INDENT_UNIT - 1)?;
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
            _ => Ok(())
        }
    }

    pub fn print_if(&mut self, test: &ast::Expr, blk: &ast::Block,
                    elseopt: Option<&ast::Expr>) -> io::Result<()> {
        self.head("if")?;
        self.print_expr(test)?;
        space(&mut self.s)?;
        self.print_block(blk)?;
        self.print_else(elseopt)
    }

    pub fn print_if_let(&mut self, pat: &ast::Pat, expr: &ast::Expr, blk: &ast::Block,
                        elseopt: Option<&ast::Expr>) -> io::Result<()> {
        self.head("if let")?;
        self.print_pat(pat)?;
        space(&mut self.s)?;
        self.word_space("=")?;
        self.print_expr(expr)?;
        space(&mut self.s)?;
        self.print_block(blk)?;
        self.print_else(elseopt)
    }

    pub fn print_mac(&mut self, m: &ast::Mac, delim: token::DelimToken)
                     -> io::Result<()> {
        self.print_path(&m.node.path, false, 0, false)?;
        word(&mut self.s, "!")?;
        match delim {
            token::Paren => self.popen()?,
            token::Bracket => word(&mut self.s, "[")?,
            token::Brace => {
                self.head("")?;
                self.bopen()?;
            }
            token::NoDelim => {}
        }
        self.print_tts(&m.node.tts)?;
        match delim {
            token::Paren => self.pclose(),
            token::Bracket => word(&mut self.s, "]"),
            token::Brace => self.bclose(m.span),
            token::NoDelim => Ok(()),
        }
    }


    fn print_call_post(&mut self, args: &[P<ast::Expr>]) -> io::Result<()> {
        self.popen()?;
        self.commasep_exprs(Inconsistent, args)?;
        self.pclose()
    }

    pub fn check_expr_bin_needs_paren(&mut self, sub_expr: &ast::Expr,
                                      binop: ast::BinOp) -> bool {
        match sub_expr.node {
            ast::ExprKind::Binary(ref sub_op, _, _) => {
                if AssocOp::from_ast_binop(sub_op.node).precedence() <
                    AssocOp::from_ast_binop(binop.node).precedence() {
                    true
                } else {
                    false
                }
            }
            _ => true
        }
    }

    pub fn print_expr_maybe_paren(&mut self, expr: &ast::Expr) -> io::Result<()> {
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

    fn print_expr_in_place(&mut self,
                           place: &ast::Expr,
                           expr: &ast::Expr) -> io::Result<()> {
        self.print_expr_maybe_paren(place)?;
        space(&mut self.s)?;
        self.word_space("<-")?;
        self.print_expr_maybe_paren(expr)
    }

    fn print_expr_vec(&mut self, exprs: &[P<ast::Expr>],
                      attrs: &[Attribute]) -> io::Result<()> {
        self.ibox(INDENT_UNIT)?;
        word(&mut self.s, "[")?;
        self.print_inner_attributes_inline(attrs)?;
        self.commasep_exprs(Inconsistent, &exprs[..])?;
        word(&mut self.s, "]")?;
        self.end()
    }

    fn print_expr_repeat(&mut self,
                         element: &ast::Expr,
                         count: &ast::Expr,
                         attrs: &[Attribute]) -> io::Result<()> {
        self.ibox(INDENT_UNIT)?;
        word(&mut self.s, "[")?;
        self.print_inner_attributes_inline(attrs)?;
        self.print_expr(element)?;
        self.word_space(";")?;
        self.print_expr(count)?;
        word(&mut self.s, "]")?;
        self.end()
    }

    fn print_expr_struct(&mut self,
                         path: &ast::Path,
                         fields: &[ast::Field],
                         wth: &Option<P<ast::Expr>>,
                         attrs: &[Attribute]) -> io::Result<()> {
        self.print_path(path, true, 0, false)?;
        word(&mut self.s, "{")?;
        self.print_inner_attributes_inline(attrs)?;
        self.commasep_cmnt(
            Consistent,
            &fields[..],
            |s, field| {
                s.ibox(INDENT_UNIT)?;
                if !field.is_shorthand {
                    s.print_ident(field.ident.node)?;
                    s.word_space(":")?;
                }
                s.print_expr(&field.expr)?;
                s.end()
            },
            |f| f.span)?;
        match *wth {
            Some(ref expr) => {
                self.ibox(INDENT_UNIT)?;
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
            }
        }
        word(&mut self.s, "}")?;
        Ok(())
    }

    fn print_expr_tup(&mut self, exprs: &[P<ast::Expr>],
                      attrs: &[Attribute]) -> io::Result<()> {
        self.popen()?;
        self.print_inner_attributes_inline(attrs)?;
        self.commasep_exprs(Inconsistent, &exprs[..])?;
        if exprs.len() == 1 {
            word(&mut self.s, ",")?;
        }
        self.pclose()
    }

    fn print_expr_call(&mut self,
                       func: &ast::Expr,
                       args: &[P<ast::Expr>]) -> io::Result<()> {
        self.print_expr_maybe_paren(func)?;
        self.print_call_post(args)
    }

    fn print_expr_method_call(&mut self,
                              ident: ast::SpannedIdent,
                              tys: &[P<ast::Ty>],
                              args: &[P<ast::Expr>]) -> io::Result<()> {
        let base_args = &args[1..];
        self.print_expr(&args[0])?;
        word(&mut self.s, ".")?;
        self.print_ident(ident.node)?;
        if !tys.is_empty() {
            word(&mut self.s, "::<")?;
            self.commasep(Inconsistent, tys,
                          |s, ty| s.print_type(&ty))?;
            word(&mut self.s, ">")?;
        }
        self.print_call_post(base_args)
    }

    fn print_expr_binary(&mut self,
                         op: ast::BinOp,
                         lhs: &ast::Expr,
                         rhs: &ast::Expr) -> io::Result<()> {
        if self.check_expr_bin_needs_paren(lhs, op) {
            self.print_expr_maybe_paren(lhs)?;
        } else {
            self.print_expr(lhs)?;
        }
        space(&mut self.s)?;
        self.word_space(op.node.to_string())?;
        if self.check_expr_bin_needs_paren(rhs, op) {
            self.print_expr_maybe_paren(rhs)
        } else {
            self.print_expr(rhs)
        }
    }

    fn print_expr_unary(&mut self,
                        op: ast::UnOp,
                        expr: &ast::Expr) -> io::Result<()> {
        word(&mut self.s, ast::UnOp::to_string(op))?;
        self.print_expr_maybe_paren(expr)
    }

    fn print_expr_addr_of(&mut self,
                          mutability: ast::Mutability,
                          expr: &ast::Expr) -> io::Result<()> {
        word(&mut self.s, "&")?;
        self.print_mutability(mutability)?;
        self.print_expr_maybe_paren(expr)
    }

    pub fn print_expr(&mut self, expr: &ast::Expr) -> io::Result<()> {
        self.print_expr_outer_attr_style(expr, true)
    }

    fn print_expr_outer_attr_style(&mut self,
                                  expr: &ast::Expr,
                                  is_inline: bool) -> io::Result<()> {
        self.maybe_print_comment(expr.span.lo)?;

        let attrs = &expr.attrs;
        if is_inline {
            self.print_outer_attributes_inline(attrs)?;
        } else {
            self.print_outer_attributes(attrs)?;
        }

        self.ibox(INDENT_UNIT)?;
        self.ann.pre(self, NodeExpr(expr))?;
        match expr.node {
            ast::ExprKind::Box(ref expr) => {
                self.word_space("box")?;
                self.print_expr(expr)?;
            }
            ast::ExprKind::InPlace(ref place, ref expr) => {
                self.print_expr_in_place(place, expr)?;
            }
            ast::ExprKind::Array(ref exprs) => {
                self.print_expr_vec(&exprs[..], attrs)?;
            }
            ast::ExprKind::Repeat(ref element, ref count) => {
                self.print_expr_repeat(&element, &count, attrs)?;
            }
            ast::ExprKind::Struct(ref path, ref fields, ref wth) => {
                self.print_expr_struct(path, &fields[..], wth, attrs)?;
            }
            ast::ExprKind::Tup(ref exprs) => {
                self.print_expr_tup(&exprs[..], attrs)?;
            }
            ast::ExprKind::Call(ref func, ref args) => {
                self.print_expr_call(&func, &args[..])?;
            }
            ast::ExprKind::MethodCall(ident, ref tys, ref args) => {
                self.print_expr_method_call(ident, &tys[..], &args[..])?;
            }
            ast::ExprKind::Binary(op, ref lhs, ref rhs) => {
                self.print_expr_binary(op, &lhs, &rhs)?;
            }
            ast::ExprKind::Unary(op, ref expr) => {
                self.print_expr_unary(op, &expr)?;
            }
            ast::ExprKind::AddrOf(m, ref expr) => {
                self.print_expr_addr_of(m, &expr)?;
            }
            ast::ExprKind::Lit(ref lit) => {
                self.print_literal(&lit)?;
            }
            ast::ExprKind::Cast(ref expr, ref ty) => {
                if let ast::ExprKind::Cast(..) = expr.node {
                    self.print_expr(&expr)?;
                } else {
                    self.print_expr_maybe_paren(&expr)?;
                }
                space(&mut self.s)?;
                self.word_space("as")?;
                self.print_type(&ty)?;
            }
            ast::ExprKind::Type(ref expr, ref ty) => {
                self.print_expr(&expr)?;
                self.word_space(":")?;
                self.print_type(&ty)?;
            }
            ast::ExprKind::If(ref test, ref blk, ref elseopt) => {
                self.print_if(&test, &blk, elseopt.as_ref().map(|e| &**e))?;
            }
            ast::ExprKind::IfLet(ref pat, ref expr, ref blk, ref elseopt) => {
                self.print_if_let(&pat, &expr, &blk, elseopt.as_ref().map(|e| &**e))?;
            }
            ast::ExprKind::While(ref test, ref blk, opt_ident) => {
                if let Some(ident) = opt_ident {
                    self.print_ident(ident.node)?;
                    self.word_space(":")?;
                }
                self.head("while")?;
                self.print_expr(&test)?;
                space(&mut self.s)?;
                self.print_block_with_attrs(&blk, attrs)?;
            }
            ast::ExprKind::WhileLet(ref pat, ref expr, ref blk, opt_ident) => {
                if let Some(ident) = opt_ident {
                    self.print_ident(ident.node)?;
                    self.word_space(":")?;
                }
                self.head("while let")?;
                self.print_pat(&pat)?;
                space(&mut self.s)?;
                self.word_space("=")?;
                self.print_expr(&expr)?;
                space(&mut self.s)?;
                self.print_block_with_attrs(&blk, attrs)?;
            }
            ast::ExprKind::ForLoop(ref pat, ref iter, ref blk, opt_ident) => {
                if let Some(ident) = opt_ident {
                    self.print_ident(ident.node)?;
                    self.word_space(":")?;
                }
                self.head("for")?;
                self.print_pat(&pat)?;
                space(&mut self.s)?;
                self.word_space("in")?;
                self.print_expr(&iter)?;
                space(&mut self.s)?;
                self.print_block_with_attrs(&blk, attrs)?;
            }
            ast::ExprKind::Loop(ref blk, opt_ident) => {
                if let Some(ident) = opt_ident {
                    self.print_ident(ident.node)?;
                    self.word_space(":")?;
                }
                self.head("loop")?;
                space(&mut self.s)?;
                self.print_block_with_attrs(&blk, attrs)?;
            }
            ast::ExprKind::Match(ref expr, ref arms) => {
                self.cbox(INDENT_UNIT)?;
                self.ibox(4)?;
                self.word_nbsp("match")?;
                self.print_expr(&expr)?;
                space(&mut self.s)?;
                self.bopen()?;
                self.print_inner_attributes_no_trailing_hardbreak(attrs)?;
                for arm in arms {
                    self.print_arm(arm)?;
                }
                self.bclose_(expr.span, INDENT_UNIT)?;
            }
            ast::ExprKind::Closure(capture_clause, ref decl, ref body, _) => {
                self.print_capture_clause(capture_clause)?;

                self.print_fn_block_args(&decl)?;
                space(&mut self.s)?;
                self.print_expr(body)?;
                self.end()?; // need to close a box

                // a box will be closed by print_expr, but we didn't want an overall
                // wrapper so we closed the corresponding opening. so create an
                // empty box to satisfy the close.
                self.ibox(0)?;
            }
            ast::ExprKind::Block(ref blk) => {
                // containing cbox, will be closed by print-block at }
                self.cbox(INDENT_UNIT)?;
                // head-box, will be closed by print-block after {
                self.ibox(0)?;
                self.print_block_with_attrs(&blk, attrs)?;
            }
            ast::ExprKind::Assign(ref lhs, ref rhs) => {
                self.print_expr(&lhs)?;
                space(&mut self.s)?;
                self.word_space("=")?;
                self.print_expr(&rhs)?;
            }
            ast::ExprKind::AssignOp(op, ref lhs, ref rhs) => {
                self.print_expr(&lhs)?;
                space(&mut self.s)?;
                word(&mut self.s, op.node.to_string())?;
                self.word_space("=")?;
                self.print_expr(&rhs)?;
            }
            ast::ExprKind::Field(ref expr, id) => {
                self.print_expr(&expr)?;
                word(&mut self.s, ".")?;
                self.print_ident(id.node)?;
            }
            ast::ExprKind::TupField(ref expr, id) => {
                self.print_expr(&expr)?;
                word(&mut self.s, ".")?;
                self.print_usize(id.node)?;
            }
            ast::ExprKind::Index(ref expr, ref index) => {
                self.print_expr(&expr)?;
                word(&mut self.s, "[")?;
                self.print_expr(&index)?;
                word(&mut self.s, "]")?;
            }
            ast::ExprKind::Range(ref start, ref end, limits) => {
                if let &Some(ref e) = start {
                    self.print_expr(&e)?;
                }
                if limits == ast::RangeLimits::HalfOpen {
                    word(&mut self.s, "..")?;
                } else {
                    word(&mut self.s, "...")?;
                }
                if let &Some(ref e) = end {
                    self.print_expr(&e)?;
                }
            }
            ast::ExprKind::Path(None, ref path) => {
                self.print_path(path, true, 0, false)?
            }
            ast::ExprKind::Path(Some(ref qself), ref path) => {
                self.print_qpath(path, qself, true)?
            }
            ast::ExprKind::Break(opt_ident, ref opt_expr) => {
                word(&mut self.s, "break")?;
                space(&mut self.s)?;
                if let Some(ident) = opt_ident {
                    self.print_ident(ident.node)?;
                    space(&mut self.s)?;
                }
                if let Some(ref expr) = *opt_expr {
                    self.print_expr(expr)?;
                    space(&mut self.s)?;
                }
            }
            ast::ExprKind::Continue(opt_ident) => {
                word(&mut self.s, "continue")?;
                space(&mut self.s)?;
                if let Some(ident) = opt_ident {
                    self.print_ident(ident.node)?;
                    space(&mut self.s)?
                }
            }
            ast::ExprKind::Ret(ref result) => {
                word(&mut self.s, "return")?;
                match *result {
                    Some(ref expr) => {
                        word(&mut self.s, " ")?;
                        self.print_expr(&expr)?;
                    }
                    _ => ()
                }
            }
            ast::ExprKind::InlineAsm(ref a) => {
                word(&mut self.s, "asm!")?;
                self.popen()?;
                self.print_string(&a.asm.as_str(), a.asm_str_style)?;
                self.word_space(":")?;

                self.commasep(Inconsistent, &a.outputs, |s, out| {
                    let constraint = out.constraint.as_str();
                    let mut ch = constraint.chars();
                    match ch.next() {
                        Some('=') if out.is_rw => {
                            s.print_string(&format!("+{}", ch.as_str()),
                                           ast::StrStyle::Cooked)?
                        }
                        _ => s.print_string(&constraint, ast::StrStyle::Cooked)?
                    }
                    s.popen()?;
                    s.print_expr(&out.expr)?;
                    s.pclose()?;
                    Ok(())
                })?;
                space(&mut self.s)?;
                self.word_space(":")?;

                self.commasep(Inconsistent, &a.inputs, |s, &(co, ref o)| {
                    s.print_string(&co.as_str(), ast::StrStyle::Cooked)?;
                    s.popen()?;
                    s.print_expr(&o)?;
                    s.pclose()?;
                    Ok(())
                })?;
                space(&mut self.s)?;
                self.word_space(":")?;

                self.commasep(Inconsistent, &a.clobbers,
                                   |s, co| {
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
                    self.commasep(Inconsistent, &options,
                                  |s, &co| {
                                      s.print_string(co, ast::StrStyle::Cooked)?;
                                      Ok(())
                                  })?;
                }

                self.pclose()?;
            }
            ast::ExprKind::Mac(ref m) => self.print_mac(m, token::Paren)?,
            ast::ExprKind::Paren(ref e) => {
                self.popen()?;
                self.print_inner_attributes_inline(attrs)?;
                self.print_expr(&e)?;
                self.pclose()?;
            },
            ast::ExprKind::Try(ref e) => {
                self.print_expr(e)?;
                word(&mut self.s, "?")?
            }
        }
        self.ann.post(self, NodeExpr(expr))?;
        self.end()
    }

    pub fn print_local_decl(&mut self, loc: &ast::Local) -> io::Result<()> {
        self.print_pat(&loc.pat)?;
        if let Some(ref ty) = loc.ty {
            self.word_space(":")?;
            self.print_type(&ty)?;
        }
        Ok(())
    }

    pub fn print_ident(&mut self, ident: ast::Ident) -> io::Result<()> {
        word(&mut self.s, &ident.name.as_str())?;
        self.ann.post(self, NodeIdent(&ident))
    }

    pub fn print_usize(&mut self, i: usize) -> io::Result<()> {
        word(&mut self.s, &i.to_string())
    }

    pub fn print_name(&mut self, name: ast::Name) -> io::Result<()> {
        word(&mut self.s, &name.as_str())?;
        self.ann.post(self, NodeName(&name))
    }

    pub fn print_for_decl(&mut self, loc: &ast::Local,
                          coll: &ast::Expr) -> io::Result<()> {
        self.print_local_decl(loc)?;
        space(&mut self.s)?;
        self.word_space("in")?;
        self.print_expr(coll)
    }

    fn print_path(&mut self,
                  path: &ast::Path,
                  colons_before_params: bool,
                  depth: usize,
                  defaults_to_global: bool)
                  -> io::Result<()>
    {
        self.maybe_print_comment(path.span.lo)?;

        let mut segments = path.segments[..path.segments.len()-depth].iter();
        if defaults_to_global && path.is_global() {
            segments.next();
        }
        for (i, segment) in segments.enumerate() {
            if i > 0 {
                word(&mut self.s, "::")?
            }
            if segment.identifier.name != keywords::CrateRoot.name() &&
               segment.identifier.name != "$crate" {
                self.print_ident(segment.identifier)?;
                if let Some(ref parameters) = segment.parameters {
                    self.print_path_parameters(parameters, colons_before_params)?;
                }
            }
        }

        Ok(())
    }

    fn print_qpath(&mut self,
                   path: &ast::Path,
                   qself: &ast::QSelf,
                   colons_before_params: bool)
                   -> io::Result<()>
    {
        word(&mut self.s, "<")?;
        self.print_type(&qself.ty)?;
        if qself.position > 0 {
            space(&mut self.s)?;
            self.word_space("as")?;
            let depth = path.segments.len() - qself.position;
            self.print_path(&path, false, depth, false)?;
        }
        word(&mut self.s, ">")?;
        word(&mut self.s, "::")?;
        let item_segment = path.segments.last().unwrap();
        self.print_ident(item_segment.identifier)?;
        match item_segment.parameters {
            Some(ref parameters) => self.print_path_parameters(parameters, colons_before_params),
            None => Ok(()),
        }
    }

    fn print_path_parameters(&mut self,
                             parameters: &ast::PathParameters,
                             colons_before_params: bool)
                             -> io::Result<()>
    {
        if colons_before_params {
            word(&mut self.s, "::")?
        }

        match *parameters {
            ast::PathParameters::AngleBracketed(ref data) => {
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
                    self.commasep(
                        Inconsistent,
                        &data.types,
                        |s, ty| s.print_type(&ty))?;
                        comma = true;
                }

                for binding in data.bindings.iter() {
                    if comma {
                        self.word_space(",")?
                    }
                    self.print_ident(binding.ident)?;
                    space(&mut self.s)?;
                    self.word_space("=")?;
                    self.print_type(&binding.ty)?;
                    comma = true;
                }

                word(&mut self.s, ">")?
            }

            ast::PathParameters::Parenthesized(ref data) => {
                word(&mut self.s, "(")?;
                self.commasep(
                    Inconsistent,
                    &data.inputs,
                    |s, ty| s.print_type(&ty))?;
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

    pub fn print_pat(&mut self, pat: &ast::Pat) -> io::Result<()> {
        self.maybe_print_comment(pat.span.lo)?;
        self.ann.pre(self, NodePat(pat))?;
        /* Pat isn't normalized, but the beauty of it
         is that it doesn't matter */
        match pat.node {
            PatKind::Wild => word(&mut self.s, "_")?,
            PatKind::Ident(binding_mode, ref path1, ref sub) => {
                match binding_mode {
                    ast::BindingMode::ByRef(mutbl) => {
                        self.word_nbsp("ref")?;
                        self.print_mutability(mutbl)?;
                    }
                    ast::BindingMode::ByValue(ast::Mutability::Immutable) => {}
                    ast::BindingMode::ByValue(ast::Mutability::Mutable) => {
                        self.word_nbsp("mut")?;
                    }
                }
                self.print_ident(path1.node)?;
                if let Some(ref p) = *sub {
                    word(&mut self.s, "@")?;
                    self.print_pat(&p)?;
                }
            }
            PatKind::TupleStruct(ref path, ref elts, ddpos) => {
                self.print_path(path, true, 0, false)?;
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
            PatKind::Path(None, ref path) => {
                self.print_path(path, true, 0, false)?;
            }
            PatKind::Path(Some(ref qself), ref path) => {
                self.print_qpath(path, qself, false)?;
            }
            PatKind::Struct(ref path, ref fields, etc) => {
                self.print_path(path, true, 0, false)?;
                self.nbsp()?;
                self.word_space("{")?;
                self.commasep_cmnt(
                    Consistent, &fields[..],
                    |s, f| {
                        s.cbox(INDENT_UNIT)?;
                        if !f.node.is_shorthand {
                            s.print_ident(f.node.ident)?;
                            s.word_nbsp(":")?;
                        }
                        s.print_pat(&f.node.pat)?;
                        s.end()
                    },
                    |f| f.node.pat.span)?;
                if etc {
                    if !fields.is_empty() { self.word_space(",")?; }
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
                if mutbl == ast::Mutability::Mutable {
                    word(&mut self.s, "mut ")?;
                }
                self.print_pat(&inner)?;
            }
            PatKind::Lit(ref e) => self.print_expr(&**e)?,
            PatKind::Range(ref begin, ref end) => {
                self.print_expr(&begin)?;
                space(&mut self.s)?;
                word(&mut self.s, "...")?;
                self.print_expr(&end)?;
            }
            PatKind::Slice(ref before, ref slice, ref after) => {
                word(&mut self.s, "[")?;
                self.commasep(Inconsistent,
                                   &before[..],
                                   |s, p| s.print_pat(&p))?;
                if let Some(ref p) = *slice {
                    if !before.is_empty() { self.word_space(",")?; }
                    if p.node != PatKind::Wild {
                        self.print_pat(&p)?;
                    }
                    word(&mut self.s, "..")?;
                    if !after.is_empty() { self.word_space(",")?; }
                }
                self.commasep(Inconsistent,
                                   &after[..],
                                   |s, p| s.print_pat(&p))?;
                word(&mut self.s, "]")?;
            }
            PatKind::Mac(ref m) => self.print_mac(m, token::Paren)?,
        }
        self.ann.post(self, NodePat(pat))
    }

    fn print_arm(&mut self, arm: &ast::Arm) -> io::Result<()> {
        // I have no idea why this check is necessary, but here it
        // is :(
        if arm.attrs.is_empty() {
            space(&mut self.s)?;
        }
        self.cbox(INDENT_UNIT)?;
        self.ibox(0)?;
        self.maybe_print_comment(arm.pats[0].span.lo)?;
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
            ast::ExprKind::Block(ref blk) => {
                // the block will close the pattern's ibox
                self.print_block_unclosed_indent(&blk, INDENT_UNIT)?;

                // If it is a user-provided unsafe block, print a comma after it
                if let BlockCheckMode::Unsafe(ast::UserProvided) = blk.rules {
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

    fn print_explicit_self(&mut self, explicit_self: &ast::ExplicitSelf) -> io::Result<()> {
        match explicit_self.node {
            SelfKind::Value(m) => {
                self.print_mutability(m)?;
                word(&mut self.s, "self")
            }
            SelfKind::Region(ref lt, m) => {
                word(&mut self.s, "&")?;
                self.print_opt_lifetime(lt)?;
                self.print_mutability(m)?;
                word(&mut self.s, "self")
            }
            SelfKind::Explicit(ref typ, m) => {
                self.print_mutability(m)?;
                word(&mut self.s, "self")?;
                self.word_space(":")?;
                self.print_type(&typ)
            }
        }
    }

    pub fn print_fn(&mut self,
                    decl: &ast::FnDecl,
                    unsafety: ast::Unsafety,
                    constness: ast::Constness,
                    abi: abi::Abi,
                    name: Option<ast::Ident>,
                    generics: &ast::Generics,
                    vis: &ast::Visibility) -> io::Result<()> {
        self.print_fn_header_info(unsafety, constness, abi, vis)?;

        if let Some(name) = name {
            self.nbsp()?;
            self.print_ident(name)?;
        }
        self.print_generics(generics)?;
        self.print_fn_args_and_ret(decl)?;
        self.print_where_clause(&generics.where_clause)
    }

    pub fn print_fn_args_and_ret(&mut self, decl: &ast::FnDecl)
        -> io::Result<()> {
        self.popen()?;
        self.commasep(Inconsistent, &decl.inputs, |s, arg| s.print_arg(arg, false))?;
        if decl.variadic {
            word(&mut self.s, ", ...")?;
        }
        self.pclose()?;

        self.print_fn_output(decl)
    }

    pub fn print_fn_block_args(
            &mut self,
            decl: &ast::FnDecl)
            -> io::Result<()> {
        word(&mut self.s, "|")?;
        self.commasep(Inconsistent, &decl.inputs, |s, arg| s.print_arg(arg, true))?;
        word(&mut self.s, "|")?;

        if let ast::FunctionRetTy::Default(..) = decl.output {
            return Ok(());
        }

        self.space_if_not_bol()?;
        self.word_space("->")?;
        match decl.output {
            ast::FunctionRetTy::Ty(ref ty) => {
                self.print_type(&ty)?;
                self.maybe_print_comment(ty.span.lo)
            }
            ast::FunctionRetTy::Default(..) => unreachable!(),
        }
    }

    pub fn print_capture_clause(&mut self, capture_clause: ast::CaptureBy)
                                -> io::Result<()> {
        match capture_clause {
            ast::CaptureBy::Value => self.word_space("move"),
            ast::CaptureBy::Ref => Ok(()),
        }
    }

    pub fn print_bounds(&mut self,
                        prefix: &str,
                        bounds: &[ast::TyParamBound])
                        -> io::Result<()> {
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

                (match *bound {
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
                })?
            }
            Ok(())
        } else {
            Ok(())
        }
    }

    pub fn print_lifetime(&mut self,
                          lifetime: &ast::Lifetime)
                          -> io::Result<()>
    {
        self.print_name(lifetime.name)
    }

    pub fn print_lifetime_bounds(&mut self,
                                 lifetime: &ast::Lifetime,
                                 bounds: &[ast::Lifetime])
                                 -> io::Result<()>
    {
        self.print_lifetime(lifetime)?;
        if !bounds.is_empty() {
            word(&mut self.s, ": ")?;
            for (i, bound) in bounds.iter().enumerate() {
                if i != 0 {
                    word(&mut self.s, " + ")?;
                }
                self.print_lifetime(bound)?;
            }
        }
        Ok(())
    }

    pub fn print_generics(&mut self,
                          generics: &ast::Generics)
                          -> io::Result<()>
    {
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
                let lifetime_def = &generics.lifetimes[idx];
                s.print_outer_attributes_inline(&lifetime_def.attrs)?;
                s.print_lifetime_bounds(&lifetime_def.lifetime, &lifetime_def.bounds)
            } else {
                let idx = idx - generics.lifetimes.len();
                let param = &generics.ty_params[idx];
                s.print_ty_param(param)
            }
        })?;

        word(&mut self.s, ">")?;
        Ok(())
    }

    pub fn print_ty_param(&mut self, param: &ast::TyParam) -> io::Result<()> {
        self.print_outer_attributes_inline(&param.attrs)?;
        self.print_ident(param.ident)?;
        self.print_bounds(":", &param.bounds)?;
        match param.default {
            Some(ref default) => {
                space(&mut self.s)?;
                self.word_space("=")?;
                self.print_type(&default)
            }
            _ => Ok(())
        }
    }

    pub fn print_where_clause(&mut self, where_clause: &ast::WhereClause)
                              -> io::Result<()> {
        if where_clause.predicates.is_empty() {
            return Ok(())
        }

        space(&mut self.s)?;
        self.word_space("where")?;

        for (i, predicate) in where_clause.predicates.iter().enumerate() {
            if i != 0 {
                self.word_space(",")?;
            }

            match *predicate {
                ast::WherePredicate::BoundPredicate(ast::WhereBoundPredicate{ref bound_lifetimes,
                                                                             ref bounded_ty,
                                                                             ref bounds,
                                                                             ..}) => {
                    self.print_formal_lifetime_list(bound_lifetimes)?;
                    self.print_type(&bounded_ty)?;
                    self.print_bounds(":", bounds)?;
                }
                ast::WherePredicate::RegionPredicate(ast::WhereRegionPredicate{ref lifetime,
                                                                               ref bounds,
                                                                               ..}) => {
                    self.print_lifetime_bounds(lifetime, bounds)?;
                }
                ast::WherePredicate::EqPredicate(ast::WhereEqPredicate{ref lhs_ty,
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

    pub fn print_view_path(&mut self, vp: &ast::ViewPath) -> io::Result<()> {
        match vp.node {
            ast::ViewPathSimple(ident, ref path) => {
                self.print_path(path, false, 0, true)?;

                if path.segments.last().unwrap().identifier.name !=
                        ident.name {
                    space(&mut self.s)?;
                    self.word_space("as")?;
                    self.print_ident(ident)?;
                }

                Ok(())
            }

            ast::ViewPathGlob(ref path) => {
                self.print_path(path, false, 0, true)?;
                word(&mut self.s, "::*")
            }

            ast::ViewPathList(ref path, ref idents) => {
                if path.segments.is_empty() {
                    word(&mut self.s, "{")?;
                } else {
                    self.print_path(path, false, 0, true)?;
                    word(&mut self.s, "::{")?;
                }
                self.commasep(Inconsistent, &idents[..], |s, w| {
                    s.print_ident(w.node.name)?;
                    if let Some(ident) = w.node.rename {
                        space(&mut s.s)?;
                        s.word_space("as")?;
                        s.print_ident(ident)?;
                    }
                    Ok(())
                })?;
                word(&mut self.s, "}")
            }
        }
    }

    pub fn print_mutability(&mut self,
                            mutbl: ast::Mutability) -> io::Result<()> {
        match mutbl {
            ast::Mutability::Mutable => self.word_nbsp("mut"),
            ast::Mutability::Immutable => Ok(()),
        }
    }

    pub fn print_mt(&mut self, mt: &ast::MutTy) -> io::Result<()> {
        self.print_mutability(mt.mutbl)?;
        self.print_type(&mt.ty)
    }

    pub fn print_arg(&mut self, input: &ast::Arg, is_closure: bool) -> io::Result<()> {
        self.ibox(INDENT_UNIT)?;
        match input.ty.node {
            ast::TyKind::Infer if is_closure => self.print_pat(&input.pat)?,
            _ => {
                if let Some(eself) = input.to_self() {
                    self.print_explicit_self(&eself)?;
                } else {
                    let invalid = if let PatKind::Ident(_, ident, _) = input.pat.node {
                        ident.node.name == keywords::Invalid.name()
                    } else {
                        false
                    };
                    if !invalid {
                        self.print_pat(&input.pat)?;
                        word(&mut self.s, ":")?;
                        space(&mut self.s)?;
                    }
                    self.print_type(&input.ty)?;
                }
            }
        }
        self.end()
    }

    pub fn print_fn_output(&mut self, decl: &ast::FnDecl) -> io::Result<()> {
        if let ast::FunctionRetTy::Default(..) = decl.output {
            return Ok(());
        }

        self.space_if_not_bol()?;
        self.ibox(INDENT_UNIT)?;
        self.word_space("->")?;
        match decl.output {
            ast::FunctionRetTy::Default(..) => unreachable!(),
            ast::FunctionRetTy::Ty(ref ty) =>
                self.print_type(&ty)?
        }
        self.end()?;

        match decl.output {
            ast::FunctionRetTy::Ty(ref output) => self.maybe_print_comment(output.span.lo),
            _ => Ok(())
        }
    }

    pub fn print_ty_fn(&mut self,
                       abi: abi::Abi,
                       unsafety: ast::Unsafety,
                       decl: &ast::FnDecl,
                       name: Option<ast::Ident>,
                       generics: &ast::Generics)
                       -> io::Result<()> {
        self.ibox(INDENT_UNIT)?;
        if !generics.lifetimes.is_empty() || !generics.ty_params.is_empty() {
            word(&mut self.s, "for")?;
            self.print_generics(generics)?;
        }
        let generics = ast::Generics {
            lifetimes: Vec::new(),
            ty_params: Vec::new(),
            where_clause: ast::WhereClause {
                id: ast::DUMMY_NODE_ID,
                predicates: Vec::new(),
            },
            span: syntax_pos::DUMMY_SP,
        };
        self.print_fn(decl,
                      unsafety,
                      ast::Constness::NotConst,
                      abi,
                      name,
                      &generics,
                      &ast::Visibility::Inherited)?;
        self.end()
    }

    pub fn maybe_print_trailing_comment(&mut self, span: syntax_pos::Span,
                                        next_pos: Option<BytePos>)
        -> io::Result<()> {
        let cm = match self.cm {
            Some(cm) => cm,
            _ => return Ok(())
        };
        if let Some(ref cmnt) = self.next_comment() {
            if cmnt.style != comments::Trailing { return Ok(()) }
            let span_line = cm.lookup_char_pos(span.hi);
            let comment_line = cm.lookup_char_pos(cmnt.pos);
            let next = next_pos.unwrap_or(cmnt.pos + BytePos(1));
            if span.hi < cmnt.pos && cmnt.pos < next && span_line.line == comment_line.line {
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
                _ => break
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
            None => Ok(())
        }
    }

    pub fn print_extern_opt_abi(&mut self,
                                opt_abi: Option<Abi>) -> io::Result<()> {
        match opt_abi {
            Some(abi) => {
                self.word_nbsp("extern")?;
                self.word_nbsp(&abi.to_string())
            }
            None => Ok(())
        }
    }

    pub fn print_fn_header_info(&mut self,
                                unsafety: ast::Unsafety,
                                constness: ast::Constness,
                                abi: Abi,
                                vis: &ast::Visibility) -> io::Result<()> {
        word(&mut self.s, &visibility_qualified(vis, ""))?;

        match constness {
            ast::Constness::NotConst => {}
            ast::Constness::Const => self.word_nbsp("const")?
        }

        self.print_unsafety(unsafety)?;

        if abi != Abi::Rust {
            self.word_nbsp("extern")?;
            self.word_nbsp(&abi.to_string())?;
        }

        word(&mut self.s, "fn")
    }

    pub fn print_unsafety(&mut self, s: ast::Unsafety) -> io::Result<()> {
        match s {
            ast::Unsafety::Normal => Ok(()),
            ast::Unsafety::Unsafe => self.word_nbsp("unsafe"),
        }
    }
}

fn repeat(s: &str, n: usize) -> String { iter::repeat(s).take(n).collect() }

#[cfg(test)]
mod tests {
    use super::*;

    use ast;
    use codemap;
    use syntax_pos;

    #[test]
    fn test_fun_to_string() {
        let abba_ident = ast::Ident::from_str("abba");

        let decl = ast::FnDecl {
            inputs: Vec::new(),
            output: ast::FunctionRetTy::Default(syntax_pos::DUMMY_SP),
            variadic: false
        };
        let generics = ast::Generics::default();
        assert_eq!(fun_to_string(&decl, ast::Unsafety::Normal,
                                 ast::Constness::NotConst,
                                 abba_ident, &generics),
                   "fn abba()");
    }

    #[test]
    fn test_variant_to_string() {
        let ident = ast::Ident::from_str("principal_skinner");

        let var = codemap::respan(syntax_pos::DUMMY_SP, ast::Variant_ {
            name: ident,
            attrs: Vec::new(),
            // making this up as I go.... ?
            data: ast::VariantData::Unit(ast::DUMMY_NODE_ID),
            disr_expr: None,
        });

        let varstr = variant_to_string(&var);
        assert_eq!(varstr, "principal_skinner");
    }
}
