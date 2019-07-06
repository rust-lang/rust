// ignore-tidy-filelength

use crate::ast::{self, BlockCheckMode, PatKind, RangeEnd, RangeSyntax};
use crate::ast::{SelfKind, GenericBound, TraitBoundModifier};
use crate::ast::{Attribute, MacDelimiter, GenericArg};
use crate::util::parser::{self, AssocOp, Fixity};
use crate::attr;
use crate::source_map::{self, SourceMap, Spanned};
use crate::parse::token::{self, BinOpToken, Nonterminal, Token, TokenKind};
use crate::parse::lexer::comments;
use crate::parse::{self, ParseSess};
use crate::print::pp::{self, Breaks};
use crate::print::pp::Breaks::{Consistent, Inconsistent};
use crate::ptr::P;
use crate::std_inject;
use crate::symbol::{kw, sym};
use crate::tokenstream::{self, TokenStream, TokenTree};

use rustc_target::spec::abi::{self, Abi};
use syntax_pos::{self, BytePos};
use syntax_pos::{DUMMY_SP, FileName};

use std::borrow::Cow;
use std::io::Read;

pub enum AnnNode<'a> {
    Ident(&'a ast::Ident),
    Name(&'a ast::Name),
    Block(&'a ast::Block),
    Item(&'a ast::Item),
    SubItem(ast::NodeId),
    Expr(&'a ast::Expr),
    Pat(&'a ast::Pat),
}

pub trait PpAnn {
    fn pre(&self, _state: &mut State<'_>, _node: AnnNode<'_>) { }
    fn post(&self, _state: &mut State<'_>, _node: AnnNode<'_>) { }
}

#[derive(Copy, Clone)]
pub struct NoAnn;

impl PpAnn for NoAnn {}

pub struct State<'a> {
    pub s: pp::Printer<'a>,
    cm: Option<&'a SourceMap>,
    comments: Option<Vec<comments::Comment>>,
    cur_cmnt: usize,
    boxes: Vec<pp::Breaks>,
    ann: &'a (dyn PpAnn+'a),
    is_expanded: bool
}

crate const INDENT_UNIT: usize = 4;

/// Requires you to pass an input filename and reader so that
/// it can scan the input text for comments to copy forward.
pub fn print_crate<'a>(cm: &'a SourceMap,
                       sess: &ParseSess,
                       krate: &ast::Crate,
                       filename: FileName,
                       input: &mut dyn Read,
                       out: &mut String,
                       ann: &'a dyn PpAnn,
                       is_expanded: bool) {
    let mut s = State::new_from_input(cm, sess, filename, input, out, ann, is_expanded);

    if is_expanded && std_inject::injected_crate_name().is_some() {
        // We need to print `#![no_std]` (and its feature gate) so that
        // compiling pretty-printed source won't inject libstd again.
        // However we don't want these attributes in the AST because
        // of the feature gate, so we fake them up here.

        // #![feature(prelude_import)]
        let pi_nested = attr::mk_nested_word_item(ast::Ident::with_empty_ctxt(sym::prelude_import));
        let list = attr::mk_list_item(
            DUMMY_SP, ast::Ident::with_empty_ctxt(sym::feature), vec![pi_nested]);
        let fake_attr = attr::mk_attr_inner(DUMMY_SP, attr::mk_attr_id(), list);
        s.print_attribute(&fake_attr);

        // #![no_std]
        let no_std_meta = attr::mk_word_item(ast::Ident::with_empty_ctxt(sym::no_std));
        let fake_attr = attr::mk_attr_inner(DUMMY_SP, attr::mk_attr_id(), no_std_meta);
        s.print_attribute(&fake_attr);
    }

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
                          ann: &'a dyn PpAnn,
                          is_expanded: bool) -> State<'a> {
        let comments = comments::gather_comments(sess, filename, input);
        State::new(cm, out, ann, Some(comments), is_expanded)
    }

    pub fn new(cm: &'a SourceMap,
               out: &'a mut String,
               ann: &'a dyn PpAnn,
               comments: Option<Vec<comments::Comment>>,
               is_expanded: bool) -> State<'a> {
        State {
            s: pp::mk_printer(out),
            cm: Some(cm),
            comments,
            cur_cmnt: 0,
            boxes: Vec::new(),
            ann,
            is_expanded,
        }
    }
}

pub fn to_string<F>(f: F) -> String where
    F: FnOnce(&mut State<'_>),
{
    let mut wr = String::new();
    {
        let mut printer = State {
            s: pp::mk_printer(&mut wr),
            cm: None,
            comments: None,
            cur_cmnt: 0,
            boxes: Vec::new(),
            ann: &NoAnn,
            is_expanded: false
        };
        f(&mut printer);
        printer.s.eof();
    }
    wr
}

fn binop_to_string(op: BinOpToken) -> &'static str {
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

pub fn literal_to_string(lit: token::Lit) -> String {
    let token::Lit { kind, symbol, suffix } = lit;
    let mut out = match kind {
        token::Byte          => format!("b'{}'", symbol),
        token::Char          => format!("'{}'", symbol),
        token::Str           => format!("\"{}\"", symbol),
        token::StrRaw(n)     => format!("r{delim}\"{string}\"{delim}",
                                        delim="#".repeat(n as usize),
                                        string=symbol),
        token::ByteStr       => format!("b\"{}\"", symbol),
        token::ByteStrRaw(n) => format!("br{delim}\"{string}\"{delim}",
                                        delim="#".repeat(n as usize),
                                        string=symbol),
        token::Integer       |
        token::Float         |
        token::Bool          |
        token::Err           => symbol.to_string(),
    };

    if let Some(suffix) = suffix {
        out.push_str(&suffix.as_str())
    }

    out
}

pub fn token_kind_to_string(tok: &TokenKind) -> String {
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
        token::DotDotEq             => "..=".to_string(),
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
        token::OpenDelim(token::NoDelim) |
        token::CloseDelim(token::NoDelim) => " ".to_string(),
        token::Pound                => "#".to_string(),
        token::Dollar               => "$".to_string(),
        token::Question             => "?".to_string(),
        token::SingleQuote          => "'".to_string(),

        /* Literals */
        token::Literal(lit) => literal_to_string(lit),

        /* Name components */
        token::Ident(s, false)      => s.to_string(),
        token::Ident(s, true)       => format!("r#{}", s),
        token::Lifetime(s)          => s.to_string(),

        /* Other */
        token::DocComment(s)        => s.to_string(),
        token::Eof                  => "<eof>".to_string(),
        token::Whitespace           => " ".to_string(),
        token::Comment              => "/* */".to_string(),
        token::Shebang(s)           => format!("/* shebang: {}*/", s),

        token::Interpolated(ref nt) => nonterminal_to_string(nt),
    }
}

pub fn token_to_string(token: &Token) -> String {
    token_kind_to_string(&token.kind)
}

crate fn nonterminal_to_string(nt: &Nonterminal) -> String {
    match *nt {
        token::NtExpr(ref e)        => expr_to_string(e),
        token::NtMeta(ref e)        => meta_item_to_string(e),
        token::NtTy(ref e)          => ty_to_string(e),
        token::NtPath(ref e)        => path_to_string(e),
        token::NtItem(ref e)        => item_to_string(e),
        token::NtBlock(ref e)       => block_to_string(e),
        token::NtStmt(ref e)        => stmt_to_string(e),
        token::NtPat(ref e)         => pat_to_string(e),
        token::NtIdent(e, false)    => ident_to_string(e),
        token::NtIdent(e, true)     => format!("r#{}", ident_to_string(e)),
        token::NtLifetime(e)        => ident_to_string(e),
        token::NtLiteral(ref e)     => expr_to_string(e),
        token::NtTT(ref tree)       => tt_to_string(tree.clone()),
        token::NtImplItem(ref e)    => impl_item_to_string(e),
        token::NtTraitItem(ref e)   => trait_item_to_string(e),
        token::NtVis(ref e)         => vis_to_string(e),
        token::NtForeignItem(ref e) => foreign_item_to_string(e),
    }
}

pub fn ty_to_string(ty: &ast::Ty) -> String {
    to_string(|s| s.print_type(ty))
}

pub fn bounds_to_string(bounds: &[ast::GenericBound]) -> String {
    to_string(|s| s.print_type_bounds("", bounds))
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

pub fn lifetime_to_string(lt: &ast::Lifetime) -> String {
    to_string(|s| s.print_lifetime(*lt))
}

pub fn tt_to_string(tt: tokenstream::TokenTree) -> String {
    to_string(|s| s.print_tt(tt))
}

pub fn tts_to_string(tts: &[tokenstream::TokenTree]) -> String {
    to_string(|s| s.print_tts(tts.iter().cloned().collect()))
}

pub fn tokens_to_string(tokens: TokenStream) -> String {
    to_string(|s| s.print_tts(tokens))
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

pub fn generic_params_to_string(generic_params: &[ast::GenericParam]) -> String {
    to_string(|s| s.print_generic_params(generic_params))
}

pub fn where_clause_to_string(i: &ast::WhereClause) -> String {
    to_string(|s| s.print_where_clause(i))
}

pub fn fn_block_to_string(p: &ast::FnDecl) -> String {
    to_string(|s| s.print_fn_block_args(p))
}

pub fn path_to_string(p: &ast::Path) -> String {
    to_string(|s| s.print_path(p, false, 0))
}

pub fn path_segment_to_string(p: &ast::PathSegment) -> String {
    to_string(|s| s.print_path_segment(p, false))
}

pub fn ident_to_string(id: ast::Ident) -> String {
    to_string(|s| s.print_ident(id))
}

pub fn vis_to_string(v: &ast::Visibility) -> String {
    to_string(|s| s.print_visibility(v))
}

pub fn fun_to_string(decl: &ast::FnDecl,
                     header: ast::FnHeader,
                     name: ast::Ident,
                     generics: &ast::Generics)
                     -> String {
    to_string(|s| {
        s.head("");
        s.print_fn(decl, header, Some(name),
                   generics, &source_map::dummy_spanned(ast::VisibilityKind::Inherited));
        s.end(); // Close the head box
        s.end(); // Close the outer box
    })
}

pub fn block_to_string(blk: &ast::Block) -> String {
    to_string(|s| {
        // containing cbox, will be closed by print-block at }
        s.cbox(INDENT_UNIT);
        // head-ibox, will be closed by print-block after {
        s.ibox(0);
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
    to_string(|s| s.print_mac(arg))
}

pub fn foreign_item_to_string(arg: &ast::ForeignItem) -> String {
    to_string(|s| s.print_foreign_item(arg))
}

pub fn visibility_qualified(vis: &ast::Visibility, s: &str) -> String {
    format!("{}{}", to_string(|s| s.print_visibility(vis)), s)
}

pub trait PrintState<'a> {
    fn writer(&mut self) -> &mut pp::Printer<'a>;
    fn boxes(&mut self) -> &mut Vec<pp::Breaks>;
    fn comments(&mut self) -> &mut Option<Vec<comments::Comment>>;
    fn cur_cmnt(&mut self) -> &mut usize;

    fn word_space<S: Into<Cow<'static, str>>>(&mut self, w: S) {
        self.writer().word(w);
        self.writer().space()
    }

    fn popen(&mut self) { self.writer().word("(") }

    fn pclose(&mut self) { self.writer().word(")") }

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

    fn hardbreak_if_not_bol(&mut self) {
        if !self.is_bol() {
            self.writer().hardbreak()
        }
    }

    // "raw box"
    fn rbox(&mut self, u: usize, b: pp::Breaks) {
        self.boxes().push(b);
        self.writer().rbox(u, b)
    }

    fn ibox(&mut self, u: usize) {
        self.boxes().push(pp::Breaks::Inconsistent);
        self.writer().ibox(u);
    }

    fn end(&mut self) {
        self.boxes().pop().unwrap();
        self.writer().end()
    }

    fn commasep<T, F>(&mut self, b: Breaks, elts: &[T], mut op: F)
        where F: FnMut(&mut Self, &T),
    {
        self.rbox(0, b);
        let mut first = true;
        for elt in elts {
            if first { first = false; } else { self.word_space(","); }
            op(self, elt);
        }
        self.end();
    }

    fn maybe_print_comment(&mut self, pos: BytePos) {
        while let Some(ref cmnt) = self.next_comment() {
            if cmnt.pos < pos {
                self.print_comment(cmnt);
            } else {
                break
            }
        }
    }

    fn print_comment(&mut self,
                     cmnt: &comments::Comment) {
        match cmnt.style {
            comments::Mixed => {
                assert_eq!(cmnt.lines.len(), 1);
                self.writer().zerobreak();
                self.writer().word(cmnt.lines[0].clone());
                self.writer().zerobreak()
            }
            comments::Isolated => {
                self.hardbreak_if_not_bol();
                for line in &cmnt.lines {
                    // Don't print empty lines because they will end up as trailing
                    // whitespace
                    if !line.is_empty() {
                        self.writer().word(line.clone());
                    }
                    self.writer().hardbreak();
                }
            }
            comments::Trailing => {
                if !self.is_bol() {
                    self.writer().word(" ");
                }
                if cmnt.lines.len() == 1 {
                    self.writer().word(cmnt.lines[0].clone());
                    self.writer().hardbreak()
                } else {
                    self.ibox(0);
                    for line in &cmnt.lines {
                        if !line.is_empty() {
                            self.writer().word(line.clone());
                        }
                        self.writer().hardbreak();
                    }
                    self.end();
                }
            }
            comments::BlankLine => {
                // We need to do at least one, possibly two hardbreaks.
                let is_semi = match self.writer().last_token() {
                    pp::Token::String(s, _) => ";" == s,
                    _ => false
                };
                if is_semi || self.is_begin() || self.is_end() {
                    self.writer().hardbreak();
                }
                self.writer().hardbreak();
            }
        }
        *self.cur_cmnt() = *self.cur_cmnt() + 1;
    }

    fn next_comment(&mut self) -> Option<comments::Comment> {
        let cur_cmnt = *self.cur_cmnt();
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

    fn print_literal(&mut self, lit: &ast::Lit) {
        self.maybe_print_comment(lit.span.lo());
        self.writer().word(literal_to_string(lit.token))
    }

    fn print_string(&mut self, st: &str,
                    style: ast::StrStyle) {
        let st = match style {
            ast::StrStyle::Cooked => {
                (format!("\"{}\"", st.escape_debug()))
            }
            ast::StrStyle::Raw(n) => {
                (format!("r{delim}\"{string}\"{delim}",
                         delim="#".repeat(n as usize),
                         string=st))
            }
        };
        self.writer().word(st)
    }

    fn print_inner_attributes(&mut self,
                              attrs: &[ast::Attribute]) {
        self.print_either_attributes(attrs, ast::AttrStyle::Inner, false, true)
    }

    fn print_inner_attributes_no_trailing_hardbreak(&mut self,
                                                   attrs: &[ast::Attribute])
                                                   {
        self.print_either_attributes(attrs, ast::AttrStyle::Inner, false, false)
    }

    fn print_outer_attributes(&mut self,
                              attrs: &[ast::Attribute]) {
        self.print_either_attributes(attrs, ast::AttrStyle::Outer, false, true)
    }

    fn print_inner_attributes_inline(&mut self,
                                     attrs: &[ast::Attribute]) {
        self.print_either_attributes(attrs, ast::AttrStyle::Inner, true, true)
    }

    fn print_outer_attributes_inline(&mut self,
                                     attrs: &[ast::Attribute]) {
        self.print_either_attributes(attrs, ast::AttrStyle::Outer, true, true)
    }

    fn print_either_attributes(&mut self,
                              attrs: &[ast::Attribute],
                              kind: ast::AttrStyle,
                              is_inline: bool,
                              trailing_hardbreak: bool) {
        let mut count = 0;
        for attr in attrs {
            if attr.style == kind {
                self.print_attribute_inline(attr, is_inline);
                if is_inline {
                    self.nbsp();
                }
                count += 1;
            }
        }
        if count > 0 && trailing_hardbreak && !is_inline {
            self.hardbreak_if_not_bol();
        }
    }

    fn print_attribute_path(&mut self, path: &ast::Path) {
        for (i, segment) in path.segments.iter().enumerate() {
            if i > 0 {
                self.writer().word("::");
            }
            if segment.ident.name != kw::PathRoot {
                if segment.ident.name == kw::DollarCrate {
                    self.print_dollar_crate(segment.ident);
                } else {
                    self.writer().word(segment.ident.as_str().to_string());
                }
            }
        }
    }

    fn print_attribute(&mut self, attr: &ast::Attribute) {
        self.print_attribute_inline(attr, false)
    }

    fn print_attribute_inline(&mut self, attr: &ast::Attribute,
                              is_inline: bool) {
        if !is_inline {
            self.hardbreak_if_not_bol();
        }
        self.maybe_print_comment(attr.span.lo());
        if attr.is_sugared_doc {
            self.writer().word(attr.value_str().unwrap().as_str().to_string());
            self.writer().hardbreak()
        } else {
            match attr.style {
                ast::AttrStyle::Inner => self.writer().word("#!["),
                ast::AttrStyle::Outer => self.writer().word("#["),
            }
            if let Some(mi) = attr.meta() {
                self.print_meta_item(&mi);
            } else {
                self.print_attribute_path(&attr.path);
                self.writer().space();
                self.print_tts(attr.tokens.clone());
            }
            self.writer().word("]");
        }
    }

    fn print_meta_list_item(&mut self, item: &ast::NestedMetaItem) {
        match item {
            ast::NestedMetaItem::MetaItem(ref mi) => {
                self.print_meta_item(mi)
            },
            ast::NestedMetaItem::Literal(ref lit) => {
                self.print_literal(lit)
            }
        }
    }

    fn print_meta_item(&mut self, item: &ast::MetaItem) {
        self.ibox(INDENT_UNIT);
        match item.node {
            ast::MetaItemKind::Word => self.print_attribute_path(&item.path),
            ast::MetaItemKind::NameValue(ref value) => {
                self.print_attribute_path(&item.path);
                self.writer().space();
                self.word_space("=");
                self.print_literal(value);
            }
            ast::MetaItemKind::List(ref items) => {
                self.print_attribute_path(&item.path);
                self.popen();
                self.commasep(Consistent,
                              &items[..],
                              |s, i| s.print_meta_list_item(i));
                self.pclose();
            }
        }
        self.end();
    }

    /// This doesn't deserve to be called "pretty" printing, but it should be
    /// meaning-preserving. A quick hack that might help would be to look at the
    /// spans embedded in the TTs to decide where to put spaces and newlines.
    /// But it'd be better to parse these according to the grammar of the
    /// appropriate macro, transcribe back into the grammar we just parsed from,
    /// and then pretty-print the resulting AST nodes (so, e.g., we print
    /// expression arguments as expressions). It can be done! I think.
    fn print_tt(&mut self, tt: tokenstream::TokenTree) {
        match tt {
            TokenTree::Token(ref token) => {
                self.writer().word(token_to_string(&token));
                match token.kind {
                    token::DocComment(..) => {
                        self.writer().hardbreak()
                    }
                    _ => {}
                }
            }
            TokenTree::Delimited(_, delim, tts) => {
                self.writer().word(token_kind_to_string(&token::OpenDelim(delim)));
                self.writer().space();
                self.print_tts(tts);
                self.writer().space();
                self.writer().word(token_kind_to_string(&token::CloseDelim(delim)))
            },
        }
    }

    fn print_tts(&mut self, tts: tokenstream::TokenStream) {
        self.ibox(0);
        for (i, tt) in tts.into_trees().enumerate() {
            if i != 0 {
                self.writer().space();
            }
            self.print_tt(tt);
        }
        self.end();
    }

    fn space_if_not_bol(&mut self) {
        if !self.is_bol() { self.writer().space(); }
    }

    fn nbsp(&mut self) { self.writer().word(" ") }

    // AST pretty-printer is used as a fallback for turning AST structures into token streams for
    // proc macros. Additionally, proc macros may stringify their input and expect it survive the
    // stringification (especially true for proc macro derives written between Rust 1.15 and 1.30).
    // So we need to somehow pretty-print `$crate` in paths in a way preserving at least some of
    // its hygiene data, most importantly name of the crate it refers to.
    // As a result we print `$crate` as `crate` if it refers to the local crate
    // and as `::other_crate_name` if it refers to some other crate.
    fn print_dollar_crate(&mut self, ident: ast::Ident) {
        let name = ident.span.ctxt().dollar_crate_name();
        if !ast::Ident::with_empty_ctxt(name).is_path_segment_keyword() {
            self.writer().word("::");
        }
        self.writer().word(name.as_str().to_string())
    }
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

impl<'a> State<'a> {
    pub fn cbox(&mut self, u: usize) {
        self.boxes.push(pp::Breaks::Consistent);
        self.s.cbox(u);
    }

    crate fn word_nbsp<S: Into<Cow<'static, str>>>(&mut self, w: S) {
        self.s.word(w);
        self.nbsp()
    }

    crate fn head<S: Into<Cow<'static, str>>>(&mut self, w: S) {
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

    crate fn bopen(&mut self) {
        self.s.word("{");
        self.end(); // close the head-box
    }

    crate fn bclose_(&mut self, span: syntax_pos::Span,
                   indented: usize) {
        self.bclose_maybe_open(span, indented, true)
    }
    crate fn bclose_maybe_open(&mut self, span: syntax_pos::Span,
                             indented: usize, close_box: bool) {
        self.maybe_print_comment(span.hi());
        self.break_offset_if_not_bol(1, -(indented as isize));
        self.s.word("}");
        if close_box {
            self.end(); // close the outer-box
        }
    }
    crate fn bclose(&mut self, span: syntax_pos::Span) {
        self.bclose_(span, INDENT_UNIT)
    }

    crate fn break_offset_if_not_bol(&mut self, n: usize,
                                   off: isize) {
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



    crate fn commasep_cmnt<T, F, G>(&mut self,
                                  b: Breaks,
                                  elts: &[T],
                                  mut op: F,
                                  mut get_span: G) where
        F: FnMut(&mut State<'_>, &T),
        G: FnMut(&T) -> syntax_pos::Span,
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
                self.maybe_print_trailing_comment(get_span(elt),
                                                  Some(get_span(&elts[i]).hi()));
                self.space_if_not_bol();
            }
        }
        self.end();
    }

    crate fn commasep_exprs(&mut self, b: Breaks,
                          exprs: &[P<ast::Expr>]) {
        self.commasep_cmnt(b, exprs, |s, e| s.print_expr(e), |e| e.span)
    }

    crate fn print_mod(&mut self, _mod: &ast::Mod,
                     attrs: &[ast::Attribute]) {
        self.print_inner_attributes(attrs);
        for item in &_mod.items {
            self.print_item(item);
        }
    }

    crate fn print_foreign_mod(&mut self, nmod: &ast::ForeignMod,
                             attrs: &[ast::Attribute]) {
        self.print_inner_attributes(attrs);
        for item in &nmod.items {
            self.print_foreign_item(item);
        }
    }

    crate fn print_opt_lifetime(&mut self, lifetime: &Option<ast::Lifetime>) {
        if let Some(lt) = *lifetime {
            self.print_lifetime(lt);
            self.nbsp();
        }
    }

    crate fn print_generic_arg(&mut self, generic_arg: &GenericArg) {
        match generic_arg {
            GenericArg::Lifetime(lt) => self.print_lifetime(*lt),
            GenericArg::Type(ty) => self.print_type(ty),
            GenericArg::Const(ct) => self.print_expr(&ct.value),
        }
    }

    crate fn print_type(&mut self, ty: &ast::Ty) {
        self.maybe_print_comment(ty.span.lo());
        self.ibox(0);
        match ty.node {
            ast::TyKind::Slice(ref ty) => {
                self.s.word("[");
                self.print_type(ty);
                self.s.word("]");
            }
            ast::TyKind::Ptr(ref mt) => {
                self.s.word("*");
                match mt.mutbl {
                    ast::Mutability::Mutable => self.word_nbsp("mut"),
                    ast::Mutability::Immutable => self.word_nbsp("const"),
                }
                self.print_type(&mt.ty);
            }
            ast::TyKind::Rptr(ref lifetime, ref mt) => {
                self.s.word("&");
                self.print_opt_lifetime(lifetime);
                self.print_mt(mt);
            }
            ast::TyKind::Never => {
                self.s.word("!");
            },
            ast::TyKind::Tup(ref elts) => {
                self.popen();
                self.commasep(Inconsistent, &elts[..],
                              |s, ty| s.print_type(ty));
                if elts.len() == 1 {
                    self.s.word(",");
                }
                self.pclose();
            }
            ast::TyKind::Paren(ref typ) => {
                self.popen();
                self.print_type(typ);
                self.pclose();
            }
            ast::TyKind::BareFn(ref f) => {
                self.print_ty_fn(f.abi,
                                 f.unsafety,
                                 &f.decl,
                                 None,
                                 &f.generic_params);
            }
            ast::TyKind::Path(None, ref path) => {
                self.print_path(path, false, 0);
            }
            ast::TyKind::Path(Some(ref qself), ref path) => {
                self.print_qpath(path, qself, false)
            }
            ast::TyKind::TraitObject(ref bounds, syntax) => {
                let prefix = if syntax == ast::TraitObjectSyntax::Dyn { "dyn" } else { "" };
                self.print_type_bounds(prefix, &bounds[..]);
            }
            ast::TyKind::ImplTrait(_, ref bounds) => {
                self.print_type_bounds("impl", &bounds[..]);
            }
            ast::TyKind::Array(ref ty, ref length) => {
                self.s.word("[");
                self.print_type(ty);
                self.s.word("; ");
                self.print_expr(&length.value);
                self.s.word("]");
            }
            ast::TyKind::Typeof(ref e) => {
                self.s.word("typeof(");
                self.print_expr(&e.value);
                self.s.word(")");
            }
            ast::TyKind::Infer => {
                self.s.word("_");
            }
            ast::TyKind::Err => {
                self.popen();
                self.s.word("/*ERROR*/");
                self.pclose();
            }
            ast::TyKind::ImplicitSelf => {
                self.s.word("Self");
            }
            ast::TyKind::Mac(ref m) => {
                self.print_mac(m);
            }
            ast::TyKind::CVarArgs => {
                self.s.word("...");
            }
        }
        self.end();
    }

    crate fn print_foreign_item(&mut self,
                              item: &ast::ForeignItem) {
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(item.span.lo());
        self.print_outer_attributes(&item.attrs);
        match item.node {
            ast::ForeignItemKind::Fn(ref decl, ref generics) => {
                self.head("");
                self.print_fn(decl, ast::FnHeader::default(),
                              Some(item.ident),
                              generics, &item.vis);
                self.end(); // end head-ibox
                self.s.word(";");
                self.end(); // end the outer fn box
            }
            ast::ForeignItemKind::Static(ref t, m) => {
                self.head(visibility_qualified(&item.vis, "static"));
                if m == ast::Mutability::Mutable {
                    self.word_space("mut");
                }
                self.print_ident(item.ident);
                self.word_space(":");
                self.print_type(t);
                self.s.word(";");
                self.end(); // end the head-ibox
                self.end(); // end the outer cbox
            }
            ast::ForeignItemKind::Ty => {
                self.head(visibility_qualified(&item.vis, "type"));
                self.print_ident(item.ident);
                self.s.word(";");
                self.end(); // end the head-ibox
                self.end(); // end the outer cbox
            }
            ast::ForeignItemKind::Macro(ref m) => {
                self.print_mac(m);
                match m.node.delim {
                    MacDelimiter::Brace => {},
                    _ => self.s.word(";")
                }
            }
        }
    }

    fn print_associated_const(&mut self,
                              ident: ast::Ident,
                              ty: &ast::Ty,
                              default: Option<&ast::Expr>,
                              vis: &ast::Visibility)
    {
        self.s.word(visibility_qualified(vis, ""));
        self.word_space("const");
        self.print_ident(ident);
        self.word_space(":");
        self.print_type(ty);
        if let Some(expr) = default {
            self.s.space();
            self.word_space("=");
            self.print_expr(expr);
        }
        self.s.word(";")
    }

    fn print_associated_type(&mut self,
                             ident: ast::Ident,
                             bounds: Option<&ast::GenericBounds>,
                             ty: Option<&ast::Ty>)
                             {
        self.word_space("type");
        self.print_ident(ident);
        if let Some(bounds) = bounds {
            self.print_type_bounds(":", bounds);
        }
        if let Some(ty) = ty {
            self.s.space();
            self.word_space("=");
            self.print_type(ty);
        }
        self.s.word(";")
    }

    /// Pretty-print an item
    crate fn print_item(&mut self, item: &ast::Item) {
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(item.span.lo());
        self.print_outer_attributes(&item.attrs);
        self.ann.pre(self, AnnNode::Item(item));
        match item.node {
            ast::ItemKind::ExternCrate(orig_name) => {
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
            ast::ItemKind::Use(ref tree) => {
                self.head(visibility_qualified(&item.vis, "use"));
                self.print_use_tree(tree);
                self.s.word(";");
                self.end(); // end inner head-block
                self.end(); // end outer head-block
            }
            ast::ItemKind::Static(ref ty, m, ref expr) => {
                self.head(visibility_qualified(&item.vis, "static"));
                if m == ast::Mutability::Mutable {
                    self.word_space("mut");
                }
                self.print_ident(item.ident);
                self.word_space(":");
                self.print_type(ty);
                self.s.space();
                self.end(); // end the head-ibox

                self.word_space("=");
                self.print_expr(expr);
                self.s.word(";");
                self.end(); // end the outer cbox
            }
            ast::ItemKind::Const(ref ty, ref expr) => {
                self.head(visibility_qualified(&item.vis, "const"));
                self.print_ident(item.ident);
                self.word_space(":");
                self.print_type(ty);
                self.s.space();
                self.end(); // end the head-ibox

                self.word_space("=");
                self.print_expr(expr);
                self.s.word(";");
                self.end(); // end the outer cbox
            }
            ast::ItemKind::Fn(ref decl, header, ref param_names, ref body) => {
                self.head("");
                self.print_fn(
                    decl,
                    header,
                    Some(item.ident),
                    param_names,
                    &item.vis
                );
                self.s.word(" ");
                self.print_block_with_attrs(body, &item.attrs);
            }
            ast::ItemKind::Mod(ref _mod) => {
                self.head(visibility_qualified(&item.vis, "mod"));
                self.print_ident(item.ident);

                if _mod.inline || self.is_expanded {
                    self.nbsp();
                    self.bopen();
                    self.print_mod(_mod, &item.attrs);
                    self.bclose(item.span);
                } else {
                    self.s.word(";");
                    self.end(); // end inner head-block
                    self.end(); // end outer head-block
                }

            }
            ast::ItemKind::ForeignMod(ref nmod) => {
                self.head("extern");
                self.word_nbsp(nmod.abi.to_string());
                self.bopen();
                self.print_foreign_mod(nmod, &item.attrs);
                self.bclose(item.span);
            }
            ast::ItemKind::GlobalAsm(ref ga) => {
                self.head(visibility_qualified(&item.vis, "global_asm!"));
                self.s.word(ga.asm.as_str().to_string());
                self.end();
            }
            ast::ItemKind::Ty(ref ty, ref generics) => {
                self.head(visibility_qualified(&item.vis, "type"));
                self.print_ident(item.ident);
                self.print_generic_params(&generics.params);
                self.end(); // end the inner ibox

                self.print_where_clause(&generics.where_clause);
                self.s.space();
                self.word_space("=");
                self.print_type(ty);
                self.s.word(";");
                self.end(); // end the outer ibox
            }
            ast::ItemKind::Existential(ref bounds, ref generics) => {
                self.head(visibility_qualified(&item.vis, "existential type"));
                self.print_ident(item.ident);
                self.print_generic_params(&generics.params);
                self.end(); // end the inner ibox

                self.print_where_clause(&generics.where_clause);
                self.s.space();
                self.print_type_bounds(":", bounds);
                self.s.word(";");
                self.end(); // end the outer ibox
            }
            ast::ItemKind::Enum(ref enum_definition, ref params) => {
                self.print_enum_def(
                    enum_definition,
                    params,
                    item.ident,
                    item.span,
                    &item.vis
                );
            }
            ast::ItemKind::Struct(ref struct_def, ref generics) => {
                self.head(visibility_qualified(&item.vis, "struct"));
                self.print_struct(struct_def, generics, item.ident, item.span, true);
            }
            ast::ItemKind::Union(ref struct_def, ref generics) => {
                self.head(visibility_qualified(&item.vis, "union"));
                self.print_struct(struct_def, generics, item.ident, item.span, true);
            }
            ast::ItemKind::Impl(unsafety,
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

                if polarity == ast::ImplPolarity::Negative {
                    self.s.word("!");
                }

                if let Some(ref t) = *opt_trait {
                    self.print_trait_ref(t);
                    self.s.space();
                    self.word_space("for");
                }

                self.print_type(ty);
                self.print_where_clause(&generics.where_clause);

                self.s.space();
                self.bopen();
                self.print_inner_attributes(&item.attrs);
                for impl_item in impl_items {
                    self.print_impl_item(impl_item);
                }
                self.bclose(item.span);
            }
            ast::ItemKind::Trait(is_auto, unsafety, ref generics, ref bounds, ref trait_items) => {
                self.head("");
                self.print_visibility(&item.vis);
                self.print_unsafety(unsafety);
                self.print_is_auto(is_auto);
                self.word_nbsp("trait");
                self.print_ident(item.ident);
                self.print_generic_params(&generics.params);
                let mut real_bounds = Vec::with_capacity(bounds.len());
                for b in bounds.iter() {
                    if let GenericBound::Trait(ref ptr, ast::TraitBoundModifier::Maybe) = *b {
                        self.s.space();
                        self.word_space("for ?");
                        self.print_trait_ref(&ptr.trait_ref);
                    } else {
                        real_bounds.push(b.clone());
                    }
                }
                self.print_type_bounds(":", &real_bounds[..]);
                self.print_where_clause(&generics.where_clause);
                self.s.word(" ");
                self.bopen();
                for trait_item in trait_items {
                    self.print_trait_item(trait_item);
                }
                self.bclose(item.span);
            }
            ast::ItemKind::TraitAlias(ref generics, ref bounds) => {
                self.head("");
                self.print_visibility(&item.vis);
                self.word_nbsp("trait");
                self.print_ident(item.ident);
                self.print_generic_params(&generics.params);
                let mut real_bounds = Vec::with_capacity(bounds.len());
                // FIXME(durka) this seems to be some quite outdated syntax
                for b in bounds.iter() {
                    if let GenericBound::Trait(ref ptr, ast::TraitBoundModifier::Maybe) = *b {
                        self.s.space();
                        self.word_space("for ?");
                        self.print_trait_ref(&ptr.trait_ref);
                    } else {
                        real_bounds.push(b.clone());
                    }
                }
                self.nbsp();
                self.print_type_bounds("=", &real_bounds[..]);
                self.print_where_clause(&generics.where_clause);
                self.s.word(";");
            }
            ast::ItemKind::Mac(ref mac) => {
                if item.ident.name == kw::Invalid {
                    self.print_mac(mac);
                    match mac.node.delim {
                        MacDelimiter::Brace => {}
                        _ => self.s.word(";"),
                    }
                } else {
                    self.print_path(&mac.node.path, false, 0);
                    self.s.word("! ");
                    self.print_ident(item.ident);
                    self.cbox(INDENT_UNIT);
                    self.popen();
                    self.print_tts(mac.node.stream());
                    self.pclose();
                    self.s.word(";");
                    self.end();
                }
            }
            ast::ItemKind::MacroDef(ref tts) => {
                self.s.word("macro_rules! ");
                self.print_ident(item.ident);
                self.cbox(INDENT_UNIT);
                self.popen();
                self.print_tts(tts.stream());
                self.pclose();
                self.s.word(";");
                self.end();
            }
        }
        self.ann.post(self, AnnNode::Item(item))
    }

    fn print_trait_ref(&mut self, t: &ast::TraitRef) {
        self.print_path(&t.path, false, 0)
    }

    fn print_formal_generic_params(
        &mut self,
        generic_params: &[ast::GenericParam]
    ) {
        if !generic_params.is_empty() {
            self.s.word("for");
            self.print_generic_params(generic_params);
            self.nbsp();
        }
    }

    fn print_poly_trait_ref(&mut self, t: &ast::PolyTraitRef) {
        self.print_formal_generic_params(&t.bound_generic_params);
        self.print_trait_ref(&t.trait_ref)
    }

    crate fn print_enum_def(&mut self, enum_definition: &ast::EnumDef,
                          generics: &ast::Generics, ident: ast::Ident,
                          span: syntax_pos::Span,
                          visibility: &ast::Visibility) {
        self.head(visibility_qualified(visibility, "enum"));
        self.print_ident(ident);
        self.print_generic_params(&generics.params);
        self.print_where_clause(&generics.where_clause);
        self.s.space();
        self.print_variants(&enum_definition.variants, span)
    }

    crate fn print_variants(&mut self,
                          variants: &[ast::Variant],
                          span: syntax_pos::Span) {
        self.bopen();
        for v in variants {
            self.space_if_not_bol();
            self.maybe_print_comment(v.span.lo());
            self.print_outer_attributes(&v.node.attrs);
            self.ibox(INDENT_UNIT);
            self.print_variant(v);
            self.s.word(",");
            self.end();
            self.maybe_print_trailing_comment(v.span, None);
        }
        self.bclose(span)
    }

    crate fn print_visibility(&mut self, vis: &ast::Visibility) {
        match vis.node {
            ast::VisibilityKind::Public => self.word_nbsp("pub"),
            ast::VisibilityKind::Crate(sugar) => match sugar {
                ast::CrateSugar::PubCrate => self.word_nbsp("pub(crate)"),
                ast::CrateSugar::JustCrate => self.word_nbsp("crate")
            }
            ast::VisibilityKind::Restricted { ref path, .. } => {
                let path = to_string(|s| s.print_path(path, false, 0));
                if path == "self" || path == "super" {
                    self.word_nbsp(format!("pub({})", path))
                } else {
                    self.word_nbsp(format!("pub(in {})", path))
                }
            }
            ast::VisibilityKind::Inherited => {}
        }
    }

    crate fn print_defaultness(&mut self, defaultness: ast::Defaultness) {
        if let ast::Defaultness::Default = defaultness {
            self.word_nbsp("default");
        }
    }

    crate fn print_struct(&mut self,
                        struct_def: &ast::VariantData,
                        generics: &ast::Generics,
                        ident: ast::Ident,
                        span: syntax_pos::Span,
                        print_finalizer: bool) {
        self.print_ident(ident);
        self.print_generic_params(&generics.params);
        match struct_def {
            ast::VariantData::Tuple(..) | ast::VariantData::Unit(..) => {
                if let ast::VariantData::Tuple(..) = struct_def {
                    self.popen();
                    self.commasep(
                        Inconsistent, struct_def.fields(),
                        |s, field| {
                            s.maybe_print_comment(field.span.lo());
                            s.print_outer_attributes(&field.attrs);
                            s.print_visibility(&field.vis);
                            s.print_type(&field.ty)
                        }
                    );
                    self.pclose();
                }
                self.print_where_clause(&generics.where_clause);
                if print_finalizer {
                    self.s.word(";");
                }
                self.end();
                self.end(); // close the outer-box
            }
            ast::VariantData::Struct(..) => {
                self.print_where_clause(&generics.where_clause);
                self.nbsp();
                self.bopen();
                self.hardbreak_if_not_bol();

                for field in struct_def.fields() {
                    self.hardbreak_if_not_bol();
                    self.maybe_print_comment(field.span.lo());
                    self.print_outer_attributes(&field.attrs);
                    self.print_visibility(&field.vis);
                    self.print_ident(field.ident.unwrap());
                    self.word_nbsp(":");
                    self.print_type(&field.ty);
                    self.s.word(",");
                }

                self.bclose(span)
            }
        }
    }

    crate fn print_variant(&mut self, v: &ast::Variant) {
        self.head("");
        let generics = ast::Generics::default();
        self.print_struct(&v.node.data, &generics, v.node.ident, v.span, false);
        match v.node.disr_expr {
            Some(ref d) => {
                self.s.space();
                self.word_space("=");
                self.print_expr(&d.value)
            }
            _ => {}
        }
    }

    crate fn print_method_sig(&mut self,
                            ident: ast::Ident,
                            generics: &ast::Generics,
                            m: &ast::MethodSig,
                            vis: &ast::Visibility)
                            {
        self.print_fn(&m.decl,
                      m.header,
                      Some(ident),
                      &generics,
                      vis)
    }

    crate fn print_trait_item(&mut self, ti: &ast::TraitItem)
                            {
        self.ann.pre(self, AnnNode::SubItem(ti.id));
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(ti.span.lo());
        self.print_outer_attributes(&ti.attrs);
        match ti.node {
            ast::TraitItemKind::Const(ref ty, ref default) => {
                self.print_associated_const(
                    ti.ident,
                    ty,
                    default.as_ref().map(|expr| &**expr),
                    &source_map::respan(ti.span.shrink_to_lo(), ast::VisibilityKind::Inherited),
                );
            }
            ast::TraitItemKind::Method(ref sig, ref body) => {
                if body.is_some() {
                    self.head("");
                }
                self.print_method_sig(
                    ti.ident,
                    &ti.generics,
                    sig,
                    &source_map::respan(ti.span.shrink_to_lo(), ast::VisibilityKind::Inherited),
                );
                if let Some(ref body) = *body {
                    self.nbsp();
                    self.print_block_with_attrs(body, &ti.attrs);
                } else {
                    self.s.word(";");
                }
            }
            ast::TraitItemKind::Type(ref bounds, ref default) => {
                self.print_associated_type(ti.ident, Some(bounds),
                                           default.as_ref().map(|ty| &**ty));
            }
            ast::TraitItemKind::Macro(ref mac) => {
                self.print_mac(mac);
                match mac.node.delim {
                    MacDelimiter::Brace => {}
                    _ => self.s.word(";"),
                }
            }
        }
        self.ann.post(self, AnnNode::SubItem(ti.id))
    }

    crate fn print_impl_item(&mut self, ii: &ast::ImplItem) {
        self.ann.pre(self, AnnNode::SubItem(ii.id));
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(ii.span.lo());
        self.print_outer_attributes(&ii.attrs);
        self.print_defaultness(ii.defaultness);
        match ii.node {
            ast::ImplItemKind::Const(ref ty, ref expr) => {
                self.print_associated_const(ii.ident, ty, Some(expr), &ii.vis);
            }
            ast::ImplItemKind::Method(ref sig, ref body) => {
                self.head("");
                self.print_method_sig(ii.ident, &ii.generics, sig, &ii.vis);
                self.nbsp();
                self.print_block_with_attrs(body, &ii.attrs);
            }
            ast::ImplItemKind::Type(ref ty) => {
                self.print_associated_type(ii.ident, None, Some(ty));
            }
            ast::ImplItemKind::Existential(ref bounds) => {
                self.word_space("existential");
                self.print_associated_type(ii.ident, Some(bounds), None);
            }
            ast::ImplItemKind::Macro(ref mac) => {
                self.print_mac(mac);
                match mac.node.delim {
                    MacDelimiter::Brace => {}
                    _ => self.s.word(";"),
                }
            }
        }
        self.ann.post(self, AnnNode::SubItem(ii.id))
    }

    crate fn print_stmt(&mut self, st: &ast::Stmt) {
        self.maybe_print_comment(st.span.lo());
        match st.node {
            ast::StmtKind::Local(ref loc) => {
                self.print_outer_attributes(&loc.attrs);
                self.space_if_not_bol();
                self.ibox(INDENT_UNIT);
                self.word_nbsp("let");

                self.ibox(INDENT_UNIT);
                self.print_local_decl(loc);
                self.end();
                if let Some(ref init) = loc.init {
                    self.nbsp();
                    self.word_space("=");
                    self.print_expr(init);
                }
                self.s.word(";");
                self.end();
            }
            ast::StmtKind::Item(ref item) => self.print_item(item),
            ast::StmtKind::Expr(ref expr) => {
                self.space_if_not_bol();
                self.print_expr_outer_attr_style(expr, false);
                if parse::classify::expr_requires_semi_to_be_stmt(expr) {
                    self.s.word(";");
                }
            }
            ast::StmtKind::Semi(ref expr) => {
                self.space_if_not_bol();
                self.print_expr_outer_attr_style(expr, false);
                self.s.word(";");
            }
            ast::StmtKind::Mac(ref mac) => {
                let (ref mac, style, ref attrs) = **mac;
                self.space_if_not_bol();
                self.print_outer_attributes(attrs);
                self.print_mac(mac);
                if style == ast::MacStmtStyle::Semicolon {
                    self.s.word(";");
                }
            }
        }
        self.maybe_print_trailing_comment(st.span, None)
    }

    crate fn print_block(&mut self, blk: &ast::Block) {
        self.print_block_with_attrs(blk, &[])
    }

    crate fn print_block_unclosed_indent(&mut self, blk: &ast::Block,
                                       indented: usize) {
        self.print_block_maybe_unclosed(blk, indented, &[], false)
    }

    crate fn print_block_with_attrs(&mut self,
                                  blk: &ast::Block,
                                  attrs: &[ast::Attribute]) {
        self.print_block_maybe_unclosed(blk, INDENT_UNIT, attrs, true)
    }

    crate fn print_block_maybe_unclosed(&mut self,
                                      blk: &ast::Block,
                                      indented: usize,
                                      attrs: &[ast::Attribute],
                                      close_box: bool) {
        match blk.rules {
            BlockCheckMode::Unsafe(..) => self.word_space("unsafe"),
            BlockCheckMode::Default => ()
        }
        self.maybe_print_comment(blk.span.lo());
        self.ann.pre(self, AnnNode::Block(blk));
        self.bopen();

        self.print_inner_attributes(attrs);

        for (i, st) in blk.stmts.iter().enumerate() {
            match st.node {
                ast::StmtKind::Expr(ref expr) if i == blk.stmts.len() - 1 => {
                    self.maybe_print_comment(st.span.lo());
                    self.space_if_not_bol();
                    self.print_expr_outer_attr_style(expr, false);
                    self.maybe_print_trailing_comment(expr.span, Some(blk.span.hi()));
                }
                _ => self.print_stmt(st),
            }
        }

        self.bclose_maybe_open(blk.span, indented, close_box);
        self.ann.post(self, AnnNode::Block(blk))
    }

    /// Print a `let pats = scrutinee` expression.
    crate fn print_let(&mut self, pats: &[P<ast::Pat>], scrutinee: &ast::Expr) {
        self.s.word("let ");

        self.print_pats(pats);
        self.s.space();

        self.word_space("=");
        self.print_expr_cond_paren(
            scrutinee,
            Self::cond_needs_par(scrutinee)
            || parser::needs_par_as_let_scrutinee(scrutinee.precedence().order())
        )
    }

    fn print_else(&mut self, els: Option<&ast::Expr>) {
        match els {
            Some(_else) => {
                match _else.node {
                    // Another `else if` block.
                    ast::ExprKind::If(ref i, ref then, ref e) => {
                        self.cbox(INDENT_UNIT - 1);
                        self.ibox(0);
                        self.s.word(" else if ");
                        self.print_expr_as_cond(i);
                        self.s.space();
                        self.print_block(then);
                        self.print_else(e.as_ref().map(|e| &**e))
                    }
                    // Final `else` block.
                    ast::ExprKind::Block(ref b, _) => {
                        self.cbox(INDENT_UNIT - 1);
                        self.ibox(0);
                        self.s.word(" else ");
                        self.print_block(b)
                    }
                    // Constraints would be great here!
                    _ => {
                        panic!("print_if saw if with weird alternative");
                    }
                }
            }
            _ => {}
        }
    }

    crate fn print_if(&mut self, test: &ast::Expr, blk: &ast::Block,
                    elseopt: Option<&ast::Expr>) {
        self.head("if");

        self.print_expr_as_cond(test);
        self.s.space();

        self.print_block(blk);
        self.print_else(elseopt)
    }

    crate fn print_mac(&mut self, m: &ast::Mac) {
        self.print_path(&m.node.path, false, 0);
        self.s.word("!");
        match m.node.delim {
            MacDelimiter::Parenthesis => self.popen(),
            MacDelimiter::Bracket => self.s.word("["),
            MacDelimiter::Brace => {
                self.head("");
                self.bopen();
            }
        }
        self.print_tts(m.node.stream());
        match m.node.delim {
            MacDelimiter::Parenthesis => self.pclose(),
            MacDelimiter::Bracket => self.s.word("]"),
            MacDelimiter::Brace => self.bclose(m.span),
        }
    }


    fn print_call_post(&mut self, args: &[P<ast::Expr>]) {
        self.popen();
        self.commasep_exprs(Inconsistent, args);
        self.pclose()
    }

    crate fn print_expr_maybe_paren(&mut self, expr: &ast::Expr, prec: i8) {
        self.print_expr_cond_paren(expr, expr.precedence().order() < prec)
    }

    /// Print an expr using syntax that's acceptable in a condition position, such as the `cond` in
    /// `if cond { ... }`.
    crate fn print_expr_as_cond(&mut self, expr: &ast::Expr) {
        self.print_expr_cond_paren(expr, Self::cond_needs_par(expr))
    }

    /// Does `expr` need parenthesis when printed in a condition position?
    fn cond_needs_par(expr: &ast::Expr) -> bool {
        match expr.node {
            // These cases need parens due to the parse error observed in #26461: `if return {}`
            // parses as the erroneous construct `if (return {})`, not `if (return) {}`.
            ast::ExprKind::Closure(..) |
            ast::ExprKind::Ret(..) |
            ast::ExprKind::Break(..) => true,

            _ => parser::contains_exterior_struct_lit(expr),
        }
    }

    /// Print `expr` or `(expr)` when `needs_par` holds.
    fn print_expr_cond_paren(&mut self, expr: &ast::Expr, needs_par: bool) {
        if needs_par {
            self.popen();
        }
        self.print_expr(expr);
        if needs_par {
            self.pclose();
        }
    }

    fn print_expr_vec(&mut self, exprs: &[P<ast::Expr>],
                      attrs: &[Attribute]) {
        self.ibox(INDENT_UNIT);
        self.s.word("[");
        self.print_inner_attributes_inline(attrs);
        self.commasep_exprs(Inconsistent, &exprs[..]);
        self.s.word("]");
        self.end();
    }

    fn print_expr_repeat(&mut self,
                         element: &ast::Expr,
                         count: &ast::AnonConst,
                         attrs: &[Attribute]) {
        self.ibox(INDENT_UNIT);
        self.s.word("[");
        self.print_inner_attributes_inline(attrs);
        self.print_expr(element);
        self.word_space(";");
        self.print_expr(&count.value);
        self.s.word("]");
        self.end();
    }

    fn print_expr_struct(&mut self,
                         path: &ast::Path,
                         fields: &[ast::Field],
                         wth: &Option<P<ast::Expr>>,
                         attrs: &[Attribute]) {
        self.print_path(path, true, 0);
        self.s.word("{");
        self.print_inner_attributes_inline(attrs);
        self.commasep_cmnt(
            Consistent,
            &fields[..],
            |s, field| {
                s.ibox(INDENT_UNIT);
                if !field.is_shorthand {
                    s.print_ident(field.ident);
                    s.word_space(":");
                }
                s.print_expr(&field.expr);
                s.end();
            },
            |f| f.span);
        match *wth {
            Some(ref expr) => {
                self.ibox(INDENT_UNIT);
                if !fields.is_empty() {
                    self.s.word(",");
                    self.s.space();
                }
                self.s.word("..");
                self.print_expr(expr);
                self.end();
            }
            _ => if !fields.is_empty() {
                self.s.word(",")
            }
        }
        self.s.word("}");
    }

    fn print_expr_tup(&mut self, exprs: &[P<ast::Expr>],
                      attrs: &[Attribute]) {
        self.popen();
        self.print_inner_attributes_inline(attrs);
        self.commasep_exprs(Inconsistent, &exprs[..]);
        if exprs.len() == 1 {
            self.s.word(",");
        }
        self.pclose()
    }

    fn print_expr_call(&mut self,
                       func: &ast::Expr,
                       args: &[P<ast::Expr>]) {
        let prec =
            match func.node {
                ast::ExprKind::Field(..) => parser::PREC_FORCE_PAREN,
                _ => parser::PREC_POSTFIX,
            };

        self.print_expr_maybe_paren(func, prec);
        self.print_call_post(args)
    }

    fn print_expr_method_call(&mut self,
                              segment: &ast::PathSegment,
                              args: &[P<ast::Expr>]) {
        let base_args = &args[1..];
        self.print_expr_maybe_paren(&args[0], parser::PREC_POSTFIX);
        self.s.word(".");
        self.print_ident(segment.ident);
        if let Some(ref args) = segment.args {
            self.print_generic_args(args, true);
        }
        self.print_call_post(base_args)
    }

    fn print_expr_binary(&mut self,
                         op: ast::BinOp,
                         lhs: &ast::Expr,
                         rhs: &ast::Expr) {
        let assoc_op = AssocOp::from_ast_binop(op.node);
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
            (&ast::ExprKind::Cast { .. }, ast::BinOpKind::Lt) |
            (&ast::ExprKind::Cast { .. }, ast::BinOpKind::Shl) => parser::PREC_FORCE_PAREN,
            // We are given `(let _ = a) OP b`.
            //
            // - When `OP <= LAnd` we should print `let _ = a OP b` to avoid redundant parens
            //   as the parser will interpret this as `(let _ = a) OP b`.
            //
            // - Otherwise, e.g. when we have `(let a = b) < c` in AST,
            //   parens are required since the parser would interpret `let a = b < c` as
            //   `let a = (b < c)`. To achieve this, we force parens.
            (&ast::ExprKind::Let { .. }, _) if !parser::needs_par_as_let_scrutinee(prec) => {
                parser::PREC_FORCE_PAREN
            }
            _ => left_prec,
        };

        self.print_expr_maybe_paren(lhs, left_prec);
        self.s.space();
        self.word_space(op.node.to_string());
        self.print_expr_maybe_paren(rhs, right_prec)
    }

    fn print_expr_unary(&mut self,
                        op: ast::UnOp,
                        expr: &ast::Expr) {
        self.s.word(ast::UnOp::to_string(op));
        self.print_expr_maybe_paren(expr, parser::PREC_PREFIX)
    }

    fn print_expr_addr_of(&mut self,
                          mutability: ast::Mutability,
                          expr: &ast::Expr) {
        self.s.word("&");
        self.print_mutability(mutability);
        self.print_expr_maybe_paren(expr, parser::PREC_PREFIX)
    }

    crate fn print_expr(&mut self, expr: &ast::Expr) {
        self.print_expr_outer_attr_style(expr, true)
    }

    fn print_expr_outer_attr_style(&mut self,
                                  expr: &ast::Expr,
                                  is_inline: bool) {
        self.maybe_print_comment(expr.span.lo());

        let attrs = &expr.attrs;
        if is_inline {
            self.print_outer_attributes_inline(attrs);
        } else {
            self.print_outer_attributes(attrs);
        }

        self.ibox(INDENT_UNIT);
        self.ann.pre(self, AnnNode::Expr(expr));
        match expr.node {
            ast::ExprKind::Box(ref expr) => {
                self.word_space("box");
                self.print_expr_maybe_paren(expr, parser::PREC_PREFIX);
            }
            ast::ExprKind::Array(ref exprs) => {
                self.print_expr_vec(&exprs[..], attrs);
            }
            ast::ExprKind::Repeat(ref element, ref count) => {
                self.print_expr_repeat(element, count, attrs);
            }
            ast::ExprKind::Struct(ref path, ref fields, ref wth) => {
                self.print_expr_struct(path, &fields[..], wth, attrs);
            }
            ast::ExprKind::Tup(ref exprs) => {
                self.print_expr_tup(&exprs[..], attrs);
            }
            ast::ExprKind::Call(ref func, ref args) => {
                self.print_expr_call(func, &args[..]);
            }
            ast::ExprKind::MethodCall(ref segment, ref args) => {
                self.print_expr_method_call(segment, &args[..]);
            }
            ast::ExprKind::Binary(op, ref lhs, ref rhs) => {
                self.print_expr_binary(op, lhs, rhs);
            }
            ast::ExprKind::Unary(op, ref expr) => {
                self.print_expr_unary(op, expr);
            }
            ast::ExprKind::AddrOf(m, ref expr) => {
                self.print_expr_addr_of(m, expr);
            }
            ast::ExprKind::Lit(ref lit) => {
                self.print_literal(lit);
            }
            ast::ExprKind::Cast(ref expr, ref ty) => {
                let prec = AssocOp::As.precedence() as i8;
                self.print_expr_maybe_paren(expr, prec);
                self.s.space();
                self.word_space("as");
                self.print_type(ty);
            }
            ast::ExprKind::Type(ref expr, ref ty) => {
                let prec = AssocOp::Colon.precedence() as i8;
                self.print_expr_maybe_paren(expr, prec);
                self.word_space(":");
                self.print_type(ty);
            }
            ast::ExprKind::Let(ref pats, ref scrutinee) => {
                self.print_let(pats, scrutinee);
            }
            ast::ExprKind::If(ref test, ref blk, ref elseopt) => {
                self.print_if(test, blk, elseopt.as_ref().map(|e| &**e));
            }
            ast::ExprKind::While(ref test, ref blk, opt_label) => {
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }
                self.head("while");
                self.print_expr_as_cond(test);
                self.s.space();
                self.print_block_with_attrs(blk, attrs);
            }
            ast::ExprKind::ForLoop(ref pat, ref iter, ref blk, opt_label) => {
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }
                self.head("for");
                self.print_pat(pat);
                self.s.space();
                self.word_space("in");
                self.print_expr_as_cond(iter);
                self.s.space();
                self.print_block_with_attrs(blk, attrs);
            }
            ast::ExprKind::Loop(ref blk, opt_label) => {
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }
                self.head("loop");
                self.s.space();
                self.print_block_with_attrs(blk, attrs);
            }
            ast::ExprKind::Match(ref expr, ref arms) => {
                self.cbox(INDENT_UNIT);
                self.ibox(4);
                self.word_nbsp("match");
                self.print_expr_as_cond(expr);
                self.s.space();
                self.bopen();
                self.print_inner_attributes_no_trailing_hardbreak(attrs);
                for arm in arms {
                    self.print_arm(arm);
                }
                self.bclose_(expr.span, INDENT_UNIT);
            }
            ast::ExprKind::Closure(
                capture_clause, asyncness, movability, ref decl, ref body, _) => {
                self.print_movability(movability);
                self.print_asyncness(asyncness);
                self.print_capture_clause(capture_clause);

                self.print_fn_block_args(decl);
                self.s.space();
                self.print_expr(body);
                self.end(); // need to close a box

                // a box will be closed by print_expr, but we didn't want an overall
                // wrapper so we closed the corresponding opening. so create an
                // empty box to satisfy the close.
                self.ibox(0);
            }
            ast::ExprKind::Block(ref blk, opt_label) => {
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }
                // containing cbox, will be closed by print-block at }
                self.cbox(INDENT_UNIT);
                // head-box, will be closed by print-block after {
                self.ibox(0);
                self.print_block_with_attrs(blk, attrs);
            }
            ast::ExprKind::Async(capture_clause, _, ref blk) => {
                self.word_nbsp("async");
                self.print_capture_clause(capture_clause);
                self.s.space();
                // cbox/ibox in analogy to the `ExprKind::Block` arm above
                self.cbox(INDENT_UNIT);
                self.ibox(0);
                self.print_block_with_attrs(blk, attrs);
            }
            ast::ExprKind::Await(origin, ref expr) => {
                match origin {
                    ast::AwaitOrigin::MacroLike => {
                        self.s.word("await!");
                        self.print_expr_maybe_paren(expr, parser::PREC_FORCE_PAREN);
                    }
                    ast::AwaitOrigin::FieldLike => {
                        self.print_expr_maybe_paren(expr, parser::PREC_POSTFIX);
                        self.s.word(".await");
                    }
                }
            }
            ast::ExprKind::Assign(ref lhs, ref rhs) => {
                let prec = AssocOp::Assign.precedence() as i8;
                self.print_expr_maybe_paren(lhs, prec + 1);
                self.s.space();
                self.word_space("=");
                self.print_expr_maybe_paren(rhs, prec);
            }
            ast::ExprKind::AssignOp(op, ref lhs, ref rhs) => {
                let prec = AssocOp::Assign.precedence() as i8;
                self.print_expr_maybe_paren(lhs, prec + 1);
                self.s.space();
                self.s.word(op.node.to_string());
                self.word_space("=");
                self.print_expr_maybe_paren(rhs, prec);
            }
            ast::ExprKind::Field(ref expr, ident) => {
                self.print_expr_maybe_paren(expr, parser::PREC_POSTFIX);
                self.s.word(".");
                self.print_ident(ident);
            }
            ast::ExprKind::Index(ref expr, ref index) => {
                self.print_expr_maybe_paren(expr, parser::PREC_POSTFIX);
                self.s.word("[");
                self.print_expr(index);
                self.s.word("]");
            }
            ast::ExprKind::Range(ref start, ref end, limits) => {
                // Special case for `Range`.  `AssocOp` claims that `Range` has higher precedence
                // than `Assign`, but `x .. x = x` gives a parse error instead of `x .. (x = x)`.
                // Here we use a fake precedence value so that any child with lower precedence than
                // a "normal" binop gets parenthesized.  (`LOr` is the lowest-precedence binop.)
                let fake_prec = AssocOp::LOr.precedence() as i8;
                if let Some(ref e) = *start {
                    self.print_expr_maybe_paren(e, fake_prec);
                }
                if limits == ast::RangeLimits::HalfOpen {
                    self.s.word("..");
                } else {
                    self.s.word("..=");
                }
                if let Some(ref e) = *end {
                    self.print_expr_maybe_paren(e, fake_prec);
                }
            }
            ast::ExprKind::Path(None, ref path) => {
                self.print_path(path, true, 0)
            }
            ast::ExprKind::Path(Some(ref qself), ref path) => {
                self.print_qpath(path, qself, true)
            }
            ast::ExprKind::Break(opt_label, ref opt_expr) => {
                self.s.word("break");
                self.s.space();
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.s.space();
                }
                if let Some(ref expr) = *opt_expr {
                    self.print_expr_maybe_paren(expr, parser::PREC_JUMP);
                    self.s.space();
                }
            }
            ast::ExprKind::Continue(opt_label) => {
                self.s.word("continue");
                self.s.space();
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.s.space()
                }
            }
            ast::ExprKind::Ret(ref result) => {
                self.s.word("return");
                if let Some(ref expr) = *result {
                    self.s.word(" ");
                    self.print_expr_maybe_paren(expr, parser::PREC_JUMP);
                }
            }
            ast::ExprKind::InlineAsm(ref a) => {
                self.s.word("asm!");
                self.popen();
                self.print_string(&a.asm.as_str(), a.asm_str_style);
                self.word_space(":");

                self.commasep(Inconsistent, &a.outputs, |s, out| {
                    let constraint = out.constraint.as_str();
                    let mut ch = constraint.chars();
                    match ch.next() {
                        Some('=') if out.is_rw => {
                            s.print_string(&format!("+{}", ch.as_str()),
                                           ast::StrStyle::Cooked)
                        }
                        _ => s.print_string(&constraint, ast::StrStyle::Cooked)
                    }
                    s.popen();
                    s.print_expr(&out.expr);
                    s.pclose();
                });
                self.s.space();
                self.word_space(":");

                self.commasep(Inconsistent, &a.inputs, |s, &(co, ref o)| {
                    s.print_string(&co.as_str(), ast::StrStyle::Cooked);
                    s.popen();
                    s.print_expr(o);
                    s.pclose();
                });
                self.s.space();
                self.word_space(":");

                self.commasep(Inconsistent, &a.clobbers,
                                   |s, co| {
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
                    self.commasep(Inconsistent, &options,
                                  |s, &co| {
                                      s.print_string(co, ast::StrStyle::Cooked);
                                  });
                }

                self.pclose();
            }
            ast::ExprKind::Mac(ref m) => self.print_mac(m),
            ast::ExprKind::Paren(ref e) => {
                self.popen();
                self.print_inner_attributes_inline(attrs);
                self.print_expr(e);
                self.pclose();
            },
            ast::ExprKind::Yield(ref e) => {
                self.s.word("yield");
                match *e {
                    Some(ref expr) => {
                        self.s.space();
                        self.print_expr_maybe_paren(expr, parser::PREC_JUMP);
                    }
                    _ => ()
                }
            }
            ast::ExprKind::Try(ref e) => {
                self.print_expr_maybe_paren(e, parser::PREC_POSTFIX);
                self.s.word("?")
            }
            ast::ExprKind::TryBlock(ref blk) => {
                self.head("try");
                self.s.space();
                self.print_block_with_attrs(blk, attrs)
            }
            ast::ExprKind::Err => {
                self.popen();
                self.s.word("/*ERROR*/");
                self.pclose()
            }
        }
        self.ann.post(self, AnnNode::Expr(expr));
        self.end();
    }

    crate fn print_local_decl(&mut self, loc: &ast::Local) {
        self.print_pat(&loc.pat);
        if let Some(ref ty) = loc.ty {
            self.word_space(":");
            self.print_type(ty);
        }
    }

    crate fn print_ident(&mut self, ident: ast::Ident) {
        if ident.is_raw_guess() {
            self.s.word(format!("r#{}", ident));
        } else {
            self.s.word(ident.as_str().to_string());
        }
        self.ann.post(self, AnnNode::Ident(&ident))
    }

    crate fn print_usize(&mut self, i: usize) {
        self.s.word(i.to_string())
    }

    crate fn print_name(&mut self, name: ast::Name) {
        self.s.word(name.as_str().to_string());
        self.ann.post(self, AnnNode::Name(&name))
    }

    fn print_path(&mut self,
                  path: &ast::Path,
                  colons_before_params: bool,
                  depth: usize) {
        self.maybe_print_comment(path.span.lo());

        for (i, segment) in path.segments[..path.segments.len() - depth].iter().enumerate() {
            if i > 0 {
                self.s.word("::")
            }
            self.print_path_segment(segment, colons_before_params);
        }
    }

    fn print_path_segment(&mut self,
                          segment: &ast::PathSegment,
                          colons_before_params: bool) {
        if segment.ident.name != kw::PathRoot {
            if segment.ident.name == kw::DollarCrate {
                self.print_dollar_crate(segment.ident);
            } else {
                self.print_ident(segment.ident);
            }
            if let Some(ref args) = segment.args {
                self.print_generic_args(args, colons_before_params);
            }
        }
    }

    fn print_qpath(&mut self,
                   path: &ast::Path,
                   qself: &ast::QSelf,
                   colons_before_params: bool)
    {
        self.s.word("<");
        self.print_type(&qself.ty);
        if qself.position > 0 {
            self.s.space();
            self.word_space("as");
            let depth = path.segments.len() - qself.position;
            self.print_path(path, false, depth);
        }
        self.s.word(">");
        self.s.word("::");
        let item_segment = path.segments.last().unwrap();
        self.print_ident(item_segment.ident);
        match item_segment.args {
            Some(ref args) => self.print_generic_args(args, colons_before_params),
            None => {},
        }
    }

    fn print_generic_args(&mut self,
                          args: &ast::GenericArgs,
                          colons_before_params: bool)
    {
        if colons_before_params {
            self.s.word("::")
        }

        match *args {
            ast::GenericArgs::AngleBracketed(ref data) => {
                self.s.word("<");

                self.commasep(Inconsistent, &data.args, |s, generic_arg| {
                    s.print_generic_arg(generic_arg)
                });

                let mut comma = data.args.len() != 0;

                for constraint in data.constraints.iter() {
                    if comma {
                        self.word_space(",")
                    }
                    self.print_ident(constraint.ident);
                    self.s.space();
                    match constraint.kind {
                        ast::AssocTyConstraintKind::Equality { ref ty } => {
                            self.word_space("=");
                            self.print_type(ty);
                        }
                        ast::AssocTyConstraintKind::Bound { ref bounds } => {
                            self.print_type_bounds(":", &*bounds);
                        }
                    }
                    comma = true;
                }

                self.s.word(">")
            }

            ast::GenericArgs::Parenthesized(ref data) => {
                self.s.word("(");
                self.commasep(
                    Inconsistent,
                    &data.inputs,
                    |s, ty| s.print_type(ty));
                self.s.word(")");

                if let Some(ref ty) = data.output {
                    self.space_if_not_bol();
                    self.word_space("->");
                    self.print_type(ty);
                }
            }
        }
    }

    crate fn print_pat(&mut self, pat: &ast::Pat) {
        self.maybe_print_comment(pat.span.lo());
        self.ann.pre(self, AnnNode::Pat(pat));
        /* Pat isn't normalized, but the beauty of it
         is that it doesn't matter */
        match pat.node {
            PatKind::Wild => self.s.word("_"),
            PatKind::Ident(binding_mode, ident, ref sub) => {
                match binding_mode {
                    ast::BindingMode::ByRef(mutbl) => {
                        self.word_nbsp("ref");
                        self.print_mutability(mutbl);
                    }
                    ast::BindingMode::ByValue(ast::Mutability::Immutable) => {}
                    ast::BindingMode::ByValue(ast::Mutability::Mutable) => {
                        self.word_nbsp("mut");
                    }
                }
                self.print_ident(ident);
                if let Some(ref p) = *sub {
                    self.s.word("@");
                    self.print_pat(p);
                }
            }
            PatKind::TupleStruct(ref path, ref elts, ddpos) => {
                self.print_path(path, true, 0);
                self.popen();
                if let Some(ddpos) = ddpos {
                    self.commasep(Inconsistent, &elts[..ddpos], |s, p| s.print_pat(p));
                    if ddpos != 0 {
                        self.word_space(",");
                    }
                    self.s.word("..");
                    if ddpos != elts.len() {
                        self.s.word(",");
                        self.commasep(Inconsistent, &elts[ddpos..], |s, p| s.print_pat(p));
                    }
                } else {
                    self.commasep(Inconsistent, &elts[..], |s, p| s.print_pat(p));
                }
                self.pclose();
            }
            PatKind::Path(None, ref path) => {
                self.print_path(path, true, 0);
            }
            PatKind::Path(Some(ref qself), ref path) => {
                self.print_qpath(path, qself, false);
            }
            PatKind::Struct(ref path, ref fields, etc) => {
                self.print_path(path, true, 0);
                self.nbsp();
                self.word_space("{");
                self.commasep_cmnt(
                    Consistent, &fields[..],
                    |s, f| {
                        s.cbox(INDENT_UNIT);
                        if !f.node.is_shorthand {
                            s.print_ident(f.node.ident);
                            s.word_nbsp(":");
                        }
                        s.print_pat(&f.node.pat);
                        s.end();
                    },
                    |f| f.node.pat.span);
                if etc {
                    if !fields.is_empty() { self.word_space(","); }
                    self.s.word("..");
                }
                self.s.space();
                self.s.word("}");
            }
            PatKind::Tuple(ref elts, ddpos) => {
                self.popen();
                if let Some(ddpos) = ddpos {
                    self.commasep(Inconsistent, &elts[..ddpos], |s, p| s.print_pat(p));
                    if ddpos != 0 {
                        self.word_space(",");
                    }
                    self.s.word("..");
                    if ddpos != elts.len() {
                        self.s.word(",");
                        self.commasep(Inconsistent, &elts[ddpos..], |s, p| s.print_pat(p));
                    }
                } else {
                    self.commasep(Inconsistent, &elts[..], |s, p| s.print_pat(p));
                    if elts.len() == 1 {
                        self.s.word(",");
                    }
                }
                self.pclose();
            }
            PatKind::Box(ref inner) => {
                self.s.word("box ");
                self.print_pat(inner);
            }
            PatKind::Ref(ref inner, mutbl) => {
                self.s.word("&");
                if mutbl == ast::Mutability::Mutable {
                    self.s.word("mut ");
                }
                self.print_pat(inner);
            }
            PatKind::Lit(ref e) => self.print_expr(&**e),
            PatKind::Range(ref begin, ref end, Spanned { node: ref end_kind, .. }) => {
                self.print_expr(begin);
                self.s.space();
                match *end_kind {
                    RangeEnd::Included(RangeSyntax::DotDotDot) => self.s.word("..."),
                    RangeEnd::Included(RangeSyntax::DotDotEq) => self.s.word("..="),
                    RangeEnd::Excluded => self.s.word(".."),
                }
                self.print_expr(end);
            }
            PatKind::Slice(ref before, ref slice, ref after) => {
                self.s.word("[");
                self.commasep(Inconsistent,
                                   &before[..],
                                   |s, p| s.print_pat(p));
                if let Some(ref p) = *slice {
                    if !before.is_empty() { self.word_space(","); }
                    if let PatKind::Wild = p.node {
                        // Print nothing
                    } else {
                        self.print_pat(p);
                    }
                    self.s.word("..");
                    if !after.is_empty() { self.word_space(","); }
                }
                self.commasep(Inconsistent,
                                   &after[..],
                                   |s, p| s.print_pat(p));
                self.s.word("]");
            }
            PatKind::Paren(ref inner) => {
                self.popen();
                self.print_pat(inner);
                self.pclose();
            }
            PatKind::Mac(ref m) => self.print_mac(m),
        }
        self.ann.post(self, AnnNode::Pat(pat))
    }

    fn print_pats(&mut self, pats: &[P<ast::Pat>]) {
        let mut first = true;
        for p in pats {
            if first {
                first = false;
            } else {
                self.s.space();
                self.word_space("|");
            }
            self.print_pat(p);
        }
    }

    fn print_arm(&mut self, arm: &ast::Arm) {
        // I have no idea why this check is necessary, but here it
        // is :(
        if arm.attrs.is_empty() {
            self.s.space();
        }
        self.cbox(INDENT_UNIT);
        self.ibox(0);
        self.maybe_print_comment(arm.pats[0].span.lo());
        self.print_outer_attributes(&arm.attrs);
        self.print_pats(&arm.pats);
        self.s.space();
        if let Some(ref e) = arm.guard {
            self.word_space("if");
            self.print_expr(e);
            self.s.space();
        }
        self.word_space("=>");

        match arm.body.node {
            ast::ExprKind::Block(ref blk, opt_label) => {
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }

                // the block will close the pattern's ibox
                self.print_block_unclosed_indent(blk, INDENT_UNIT);

                // If it is a user-provided unsafe block, print a comma after it
                if let BlockCheckMode::Unsafe(ast::UserProvided) = blk.rules {
                    self.s.word(",");
                }
            }
            _ => {
                self.end(); // close the ibox for the pattern
                self.print_expr(&arm.body);
                self.s.word(",");
            }
        }
        self.end(); // close enclosing cbox
    }

    fn print_explicit_self(&mut self, explicit_self: &ast::ExplicitSelf) {
        match explicit_self.node {
            SelfKind::Value(m) => {
                self.print_mutability(m);
                self.s.word("self")
            }
            SelfKind::Region(ref lt, m) => {
                self.s.word("&");
                self.print_opt_lifetime(lt);
                self.print_mutability(m);
                self.s.word("self")
            }
            SelfKind::Explicit(ref typ, m) => {
                self.print_mutability(m);
                self.s.word("self");
                self.word_space(":");
                self.print_type(typ)
            }
        }
    }

    crate fn print_fn(&mut self,
                    decl: &ast::FnDecl,
                    header: ast::FnHeader,
                    name: Option<ast::Ident>,
                    generics: &ast::Generics,
                    vis: &ast::Visibility) {
        self.print_fn_header_info(header, vis);

        if let Some(name) = name {
            self.nbsp();
            self.print_ident(name);
        }
        self.print_generic_params(&generics.params);
        self.print_fn_args_and_ret(decl);
        self.print_where_clause(&generics.where_clause)
    }

    crate fn print_fn_args_and_ret(&mut self, decl: &ast::FnDecl) {
        self.popen();
        self.commasep(Inconsistent, &decl.inputs, |s, arg| s.print_arg(arg, false));
        self.pclose();

        self.print_fn_output(decl)
    }

    crate fn print_fn_block_args(&mut self, decl: &ast::FnDecl) {
        self.s.word("|");
        self.commasep(Inconsistent, &decl.inputs, |s, arg| s.print_arg(arg, true));
        self.s.word("|");

        if let ast::FunctionRetTy::Default(..) = decl.output {
            return;
        }

        self.space_if_not_bol();
        self.word_space("->");
        match decl.output {
            ast::FunctionRetTy::Ty(ref ty) => {
                self.print_type(ty);
                self.maybe_print_comment(ty.span.lo())
            }
            ast::FunctionRetTy::Default(..) => unreachable!(),
        }
    }

    crate fn print_movability(&mut self, movability: ast::Movability) {
        match movability {
            ast::Movability::Static => self.word_space("static"),
            ast::Movability::Movable => {},
        }
    }

    crate fn print_asyncness(&mut self, asyncness: ast::IsAsync) {
        if asyncness.is_async() {
            self.word_nbsp("async");
        }
    }

    crate fn print_capture_clause(&mut self, capture_clause: ast::CaptureBy) {
        match capture_clause {
            ast::CaptureBy::Value => self.word_space("move"),
            ast::CaptureBy::Ref => {},
        }
    }

    crate fn print_type_bounds(&mut self, prefix: &'static str, bounds: &[ast::GenericBound]) {
        if !bounds.is_empty() {
            self.s.word(prefix);
            let mut first = true;
            for bound in bounds {
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
                    GenericBound::Outlives(lt) => self.print_lifetime(*lt),
                }
            }
        }
    }

    crate fn print_lifetime(&mut self, lifetime: ast::Lifetime) {
        self.print_name(lifetime.ident.name)
    }

    crate fn print_lifetime_bounds(
        &mut self, lifetime: ast::Lifetime, bounds: &ast::GenericBounds) {
        self.print_lifetime(lifetime);
        if !bounds.is_empty() {
            self.s.word(": ");
            for (i, bound) in bounds.iter().enumerate() {
                if i != 0 {
                    self.s.word(" + ");
                }
                match bound {
                    ast::GenericBound::Outlives(lt) => self.print_lifetime(*lt),
                    _ => panic!(),
                }
            }
        }
    }

    crate fn print_generic_params(&mut self, generic_params: &[ast::GenericParam]) {
        if generic_params.is_empty() {
            return;
        }

        self.s.word("<");

        self.commasep(Inconsistent, &generic_params, |s, param| {
            match param.kind {
                ast::GenericParamKind::Lifetime => {
                    s.print_outer_attributes_inline(&param.attrs);
                    let lt = ast::Lifetime { id: param.id, ident: param.ident };
                    s.print_lifetime_bounds(lt, &param.bounds)
                }
                ast::GenericParamKind::Type { ref default } => {
                    s.print_outer_attributes_inline(&param.attrs);
                    s.print_ident(param.ident);
                    s.print_type_bounds(":", &param.bounds);
                    match default {
                        Some(ref default) => {
                            s.s.space();
                            s.word_space("=");
                            s.print_type(default)
                        }
                        _ => {}
                    }
                }
                ast::GenericParamKind::Const { ref ty } => {
                    s.print_outer_attributes_inline(&param.attrs);
                    s.word_space("const");
                    s.print_ident(param.ident);
                    s.s.space();
                    s.word_space(":");
                    s.print_type(ty);
                    s.print_type_bounds(":", &param.bounds)
                }
            }
        });

        self.s.word(">");
    }

    crate fn print_where_clause(&mut self, where_clause: &ast::WhereClause) {
        if where_clause.predicates.is_empty() {
            return;
        }

        self.s.space();
        self.word_space("where");

        for (i, predicate) in where_clause.predicates.iter().enumerate() {
            if i != 0 {
                self.word_space(",");
            }

            match *predicate {
                ast::WherePredicate::BoundPredicate(ast::WhereBoundPredicate {
                    ref bound_generic_params,
                    ref bounded_ty,
                    ref bounds,
                    ..
                }) => {
                    self.print_formal_generic_params(bound_generic_params);
                    self.print_type(bounded_ty);
                    self.print_type_bounds(":", bounds);
                }
                ast::WherePredicate::RegionPredicate(ast::WhereRegionPredicate{ref lifetime,
                                                                               ref bounds,
                                                                               ..}) => {
                    self.print_lifetime_bounds(*lifetime, bounds);
                }
                ast::WherePredicate::EqPredicate(ast::WhereEqPredicate{ref lhs_ty,
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

    crate fn print_use_tree(&mut self, tree: &ast::UseTree) {
        match tree.kind {
            ast::UseTreeKind::Simple(rename, ..) => {
                self.print_path(&tree.prefix, false, 0);
                if let Some(rename) = rename {
                    self.s.space();
                    self.word_space("as");
                    self.print_ident(rename);
                }
            }
            ast::UseTreeKind::Glob => {
                if !tree.prefix.segments.is_empty() {
                    self.print_path(&tree.prefix, false, 0);
                    self.s.word("::");
                }
                self.s.word("*");
            }
            ast::UseTreeKind::Nested(ref items) => {
                if tree.prefix.segments.is_empty() {
                    self.s.word("{");
                } else {
                    self.print_path(&tree.prefix, false, 0);
                    self.s.word("::{");
                }
                self.commasep(Inconsistent, &items[..], |this, &(ref tree, _)| {
                    this.print_use_tree(tree)
                });
                self.s.word("}");
            }
        }
    }

    crate fn print_mutability(&mut self, mutbl: ast::Mutability) {
        match mutbl {
            ast::Mutability::Mutable => self.word_nbsp("mut"),
            ast::Mutability::Immutable => {},
        }
    }

    crate fn print_mt(&mut self, mt: &ast::MutTy) {
        self.print_mutability(mt.mutbl);
        self.print_type(&mt.ty)
    }

    crate fn print_arg(&mut self, input: &ast::Arg, is_closure: bool) {
        self.ibox(INDENT_UNIT);
        match input.ty.node {
            ast::TyKind::Infer if is_closure => self.print_pat(&input.pat),
            _ => {
                if let Some(eself) = input.to_self() {
                    self.print_explicit_self(&eself);
                } else {
                    let invalid = if let PatKind::Ident(_, ident, _) = input.pat.node {
                        ident.name == kw::Invalid
                    } else {
                        false
                    };
                    if !invalid {
                        self.print_pat(&input.pat);
                        self.s.word(":");
                        self.s.space();
                    }
                    self.print_type(&input.ty);
                }
            }
        }
        self.end();
    }

    crate fn print_fn_output(&mut self, decl: &ast::FnDecl) {
        if let ast::FunctionRetTy::Default(..) = decl.output {
            return;
        }

        self.space_if_not_bol();
        self.ibox(INDENT_UNIT);
        self.word_space("->");
        match decl.output {
            ast::FunctionRetTy::Default(..) => unreachable!(),
            ast::FunctionRetTy::Ty(ref ty) =>
                self.print_type(ty),
        }
        self.end();

        match decl.output {
            ast::FunctionRetTy::Ty(ref output) => self.maybe_print_comment(output.span.lo()),
            _ => {}
        }
    }

    crate fn print_ty_fn(&mut self,
                       abi: abi::Abi,
                       unsafety: ast::Unsafety,
                       decl: &ast::FnDecl,
                       name: Option<ast::Ident>,
                       generic_params: &[ast::GenericParam])
                       {
        self.ibox(INDENT_UNIT);
        if !generic_params.is_empty() {
            self.s.word("for");
            self.print_generic_params(generic_params);
        }
        let generics = ast::Generics {
            params: Vec::new(),
            where_clause: ast::WhereClause {
                predicates: Vec::new(),
                span: syntax_pos::DUMMY_SP,
            },
            span: syntax_pos::DUMMY_SP,
        };
        self.print_fn(decl,
                      ast::FnHeader { unsafety, abi, ..ast::FnHeader::default() },
                      name,
                      &generics,
                      &source_map::dummy_spanned(ast::VisibilityKind::Inherited));
        self.end();
    }

    crate fn maybe_print_trailing_comment(&mut self, span: syntax_pos::Span,
                                        next_pos: Option<BytePos>)
        {
        let cm = match self.cm {
            Some(cm) => cm,
            _ => return,
        };
        if let Some(ref cmnt) = self.next_comment() {
            if cmnt.style != comments::Trailing { return; }
            let span_line = cm.lookup_char_pos(span.hi());
            let comment_line = cm.lookup_char_pos(cmnt.pos);
            let next = next_pos.unwrap_or_else(|| cmnt.pos + BytePos(1));
            if span.hi() < cmnt.pos && cmnt.pos < next && span_line.line == comment_line.line {
                self.print_comment(cmnt);
            }
        }
    }

    crate fn print_remaining_comments(&mut self) {
        // If there aren't any remaining comments, then we need to manually
        // make sure there is a line break at the end.
        if self.next_comment().is_none() {
            self.s.hardbreak();
        }
        while let Some(ref cmnt) = self.next_comment() {
            self.print_comment(cmnt);
        }
    }

    crate fn print_fn_header_info(&mut self,
                                header: ast::FnHeader,
                                vis: &ast::Visibility) {
        self.s.word(visibility_qualified(vis, ""));

        match header.constness.node {
            ast::Constness::NotConst => {}
            ast::Constness::Const => self.word_nbsp("const")
        }

        self.print_asyncness(header.asyncness.node);
        self.print_unsafety(header.unsafety);

        if header.abi != Abi::Rust {
            self.word_nbsp("extern");
            self.word_nbsp(header.abi.to_string());
        }

        self.s.word("fn")
    }

    crate fn print_unsafety(&mut self, s: ast::Unsafety) {
        match s {
            ast::Unsafety::Normal => {},
            ast::Unsafety::Unsafe => self.word_nbsp("unsafe"),
        }
    }

    crate fn print_is_auto(&mut self, s: ast::IsAuto) {
        match s {
            ast::IsAuto::Yes => self.word_nbsp("auto"),
            ast::IsAuto::No => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ast;
    use crate::source_map;
    use crate::with_default_globals;
    use syntax_pos;

    #[test]
    fn test_fun_to_string() {
        with_default_globals(|| {
            let abba_ident = ast::Ident::from_str("abba");

            let decl = ast::FnDecl {
                inputs: Vec::new(),
                output: ast::FunctionRetTy::Default(syntax_pos::DUMMY_SP),
                c_variadic: false
            };
            let generics = ast::Generics::default();
            assert_eq!(
                fun_to_string(
                    &decl,
                    ast::FnHeader {
                        unsafety: ast::Unsafety::Normal,
                        constness: source_map::dummy_spanned(ast::Constness::NotConst),
                        asyncness: source_map::dummy_spanned(ast::IsAsync::NotAsync),
                        abi: Abi::Rust,
                    },
                    abba_ident,
                    &generics
                ),
                "fn abba()"
            );
        })
    }

    #[test]
    fn test_variant_to_string() {
        with_default_globals(|| {
            let ident = ast::Ident::from_str("principal_skinner");

            let var = source_map::respan(syntax_pos::DUMMY_SP, ast::Variant_ {
                ident,
                attrs: Vec::new(),
                id: ast::DUMMY_NODE_ID,
                // making this up as I go.... ?
                data: ast::VariantData::Unit(ast::DUMMY_NODE_ID),
                disr_expr: None,
            });

            let varstr = variant_to_string(&var);
            assert_eq!(varstr, "principal_skinner");
        })
    }
}
