use crate::pp::Breaks::{Consistent, Inconsistent};
use crate::pp::{self, Breaks};

use rustc_ast::attr;
use rustc_ast::ptr::P;
use rustc_ast::token::{self, BinOpToken, CommentKind, DelimToken, Nonterminal, Token, TokenKind};
use rustc_ast::tokenstream::{TokenStream, TokenTree};
use rustc_ast::util::classify;
use rustc_ast::util::comments::{gather_comments, Comment, CommentStyle};
use rustc_ast::util::parser::{self, AssocOp, Fixity};
use rustc_ast::{self as ast, BlockCheckMode, PatKind, RangeEnd, RangeSyntax};
use rustc_ast::{GenericArg, MacArgs};
use rustc_ast::{GenericBound, SelfKind, TraitBoundModifier};
use rustc_ast::{InlineAsmOperand, InlineAsmRegOrRegClass};
use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_span::edition::Edition;
use rustc_span::source_map::{SourceMap, Spanned};
use rustc_span::symbol::{kw, sym, Ident, IdentPrinter, Symbol};
use rustc_span::{BytePos, FileName, Span};

use std::borrow::Cow;

pub enum MacHeader<'a> {
    Path(&'a ast::Path),
    Keyword(&'static str),
}

pub enum AnnNode<'a> {
    Ident(&'a Ident),
    Name(&'a Symbol),
    Block(&'a ast::Block),
    Item(&'a ast::Item),
    SubItem(ast::NodeId),
    Expr(&'a ast::Expr),
    Pat(&'a ast::Pat),
    Crate(&'a ast::Crate),
}

pub trait PpAnn {
    fn pre(&self, _state: &mut State<'_>, _node: AnnNode<'_>) {}
    fn post(&self, _state: &mut State<'_>, _node: AnnNode<'_>) {}
}

#[derive(Copy, Clone)]
pub struct NoAnn;

impl PpAnn for NoAnn {}

pub struct Comments<'a> {
    sm: &'a SourceMap,
    comments: Vec<Comment>,
    current: usize,
}

impl<'a> Comments<'a> {
    pub fn new(sm: &'a SourceMap, filename: FileName, input: String) -> Comments<'a> {
        let comments = gather_comments(sm, filename, input);
        Comments { sm, comments, current: 0 }
    }

    pub fn next(&self) -> Option<Comment> {
        self.comments.get(self.current).cloned()
    }

    pub fn trailing_comment(
        &self,
        span: rustc_span::Span,
        next_pos: Option<BytePos>,
    ) -> Option<Comment> {
        if let Some(cmnt) = self.next() {
            if cmnt.style != CommentStyle::Trailing {
                return None;
            }
            let span_line = self.sm.lookup_char_pos(span.hi());
            let comment_line = self.sm.lookup_char_pos(cmnt.pos);
            let next = next_pos.unwrap_or_else(|| cmnt.pos + BytePos(1));
            if span.hi() < cmnt.pos && cmnt.pos < next && span_line.line == comment_line.line {
                return Some(cmnt);
            }
        }

        None
    }
}

pub struct State<'a> {
    pub s: pp::Printer,
    comments: Option<Comments<'a>>,
    ann: &'a (dyn PpAnn + 'a),
    is_expanded: bool,
    // If `true`, additional parenthesis (separate from `ExprKind::Paren`)
    // are inserted to ensure that proper precedence is preserved
    // in the pretty-printed output.
    //
    // This is usually `true`, except when performing the pretty-print/reparse
    // check in `nt_to_tokenstream`
    insert_extra_parens: bool,
}

crate const INDENT_UNIT: usize = 4;

/// Requires you to pass an input filename and reader so that
/// it can scan the input text for comments to copy forward.
pub fn print_crate<'a>(
    sm: &'a SourceMap,
    krate: &ast::Crate,
    filename: FileName,
    input: String,
    ann: &'a dyn PpAnn,
    is_expanded: bool,
    edition: Edition,
) -> String {
    let mut s = State {
        s: pp::mk_printer(),
        comments: Some(Comments::new(sm, filename, input)),
        ann,
        is_expanded,
        insert_extra_parens: true,
    };

    if is_expanded && !krate.attrs.iter().any(|attr| attr.has_name(sym::no_core)) {
        // We need to print `#![no_std]` (and its feature gate) so that
        // compiling pretty-printed source won't inject libstd again.
        // However, we don't want these attributes in the AST because
        // of the feature gate, so we fake them up here.

        // `#![feature(prelude_import)]`
        let pi_nested = attr::mk_nested_word_item(Ident::with_dummy_span(sym::prelude_import));
        let list = attr::mk_list_item(Ident::with_dummy_span(sym::feature), vec![pi_nested]);
        let fake_attr = attr::mk_attr_inner(list);
        s.print_attribute(&fake_attr);

        // Currently, in Rust 2018 we don't have `extern crate std;` at the crate
        // root, so this is not needed, and actually breaks things.
        if edition == Edition::Edition2015 {
            // `#![no_std]`
            let no_std_meta = attr::mk_word_item(Ident::with_dummy_span(sym::no_std));
            let fake_attr = attr::mk_attr_inner(no_std_meta);
            s.print_attribute(&fake_attr);
        }
    }

    s.print_mod(&krate.module, &krate.attrs);
    s.print_remaining_comments();
    s.ann.post(&mut s, AnnNode::Crate(krate));
    s.s.eof()
}

// This makes printed token streams look slightly nicer,
// and also addresses some specific regressions described in #63896 and #73345.
fn tt_prepend_space(tt: &TokenTree, prev: &TokenTree) -> bool {
    if let TokenTree::Token(token) = prev {
        if let token::DocComment(comment_kind, ..) = token.kind {
            return comment_kind != CommentKind::Line;
        }
    }
    match tt {
        TokenTree::Token(token) => token.kind != token::Comma,
        TokenTree::Delimited(_, DelimToken::Paren, _) => {
            !matches!(prev, TokenTree::Token(Token { kind: token::Ident(..), .. }))
        }
        TokenTree::Delimited(_, DelimToken::Bracket, _) => {
            !matches!(prev, TokenTree::Token(Token { kind: token::Pound, .. }))
        }
        TokenTree::Delimited(..) => true,
    }
}

fn binop_to_string(op: BinOpToken) -> &'static str {
    match op {
        token::Plus => "+",
        token::Minus => "-",
        token::Star => "*",
        token::Slash => "/",
        token::Percent => "%",
        token::Caret => "^",
        token::And => "&",
        token::Or => "|",
        token::Shl => "<<",
        token::Shr => ">>",
    }
}

fn doc_comment_to_string(
    comment_kind: CommentKind,
    attr_style: ast::AttrStyle,
    data: Symbol,
) -> String {
    match (comment_kind, attr_style) {
        (CommentKind::Line, ast::AttrStyle::Outer) => format!("///{}", data),
        (CommentKind::Line, ast::AttrStyle::Inner) => format!("//!{}", data),
        (CommentKind::Block, ast::AttrStyle::Outer) => format!("/**{}*/", data),
        (CommentKind::Block, ast::AttrStyle::Inner) => format!("/*!{}*/", data),
    }
}

pub fn literal_to_string(lit: token::Lit) -> String {
    let token::Lit { kind, symbol, suffix } = lit;
    let mut out = match kind {
        token::Byte => format!("b'{}'", symbol),
        token::Char => format!("'{}'", symbol),
        token::Str => format!("\"{}\"", symbol),
        token::StrRaw(n) => {
            format!("r{delim}\"{string}\"{delim}", delim = "#".repeat(n as usize), string = symbol)
        }
        token::ByteStr => format!("b\"{}\"", symbol),
        token::ByteStrRaw(n) => {
            format!("br{delim}\"{string}\"{delim}", delim = "#".repeat(n as usize), string = symbol)
        }
        token::Integer | token::Float | token::Bool | token::Err => symbol.to_string(),
    };

    if let Some(suffix) = suffix {
        out.push_str(&suffix.as_str())
    }

    out
}

fn visibility_qualified(vis: &ast::Visibility, s: &str) -> String {
    format!("{}{}", State::new().to_string(|s| s.print_visibility(vis)), s)
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

pub trait PrintState<'a>: std::ops::Deref<Target = pp::Printer> + std::ops::DerefMut {
    fn insert_extra_parens(&self) -> bool;
    fn comments(&mut self) -> &mut Option<Comments<'a>>;
    fn print_ident(&mut self, ident: Ident);
    fn print_generic_args(&mut self, args: &ast::GenericArgs, colons_before_params: bool);

    fn strsep<T, F>(
        &mut self,
        sep: &'static str,
        space_before: bool,
        b: Breaks,
        elts: &[T],
        mut op: F,
    ) where
        F: FnMut(&mut Self, &T),
    {
        self.rbox(0, b);
        if let Some((first, rest)) = elts.split_first() {
            op(self, first);
            for elt in rest {
                if space_before {
                    self.space();
                }
                self.word_space(sep);
                op(self, elt);
            }
        }
        self.end();
    }

    fn commasep<T, F>(&mut self, b: Breaks, elts: &[T], op: F)
    where
        F: FnMut(&mut Self, &T),
    {
        self.strsep(",", false, b, elts, op)
    }

    fn maybe_print_comment(&mut self, pos: BytePos) {
        while let Some(ref cmnt) = self.next_comment() {
            if cmnt.pos < pos {
                self.print_comment(cmnt);
            } else {
                break;
            }
        }
    }

    fn print_comment(&mut self, cmnt: &Comment) {
        match cmnt.style {
            CommentStyle::Mixed => {
                if !self.is_beginning_of_line() {
                    self.zerobreak();
                }
                if let Some((last, lines)) = cmnt.lines.split_last() {
                    self.ibox(0);

                    for line in lines {
                        self.word(line.clone());
                        self.hardbreak()
                    }

                    self.word(last.clone());
                    self.space();

                    self.end();
                }
                self.zerobreak()
            }
            CommentStyle::Isolated => {
                self.hardbreak_if_not_bol();
                for line in &cmnt.lines {
                    // Don't print empty lines because they will end up as trailing
                    // whitespace.
                    if !line.is_empty() {
                        self.word(line.clone());
                    }
                    self.hardbreak();
                }
            }
            CommentStyle::Trailing => {
                if !self.is_beginning_of_line() {
                    self.word(" ");
                }
                if cmnt.lines.len() == 1 {
                    self.word(cmnt.lines[0].clone());
                    self.hardbreak()
                } else {
                    self.ibox(0);
                    for line in &cmnt.lines {
                        if !line.is_empty() {
                            self.word(line.clone());
                        }
                        self.hardbreak();
                    }
                    self.end();
                }
            }
            CommentStyle::BlankLine => {
                // We need to do at least one, possibly two hardbreaks.
                let twice = match self.last_token() {
                    pp::Token::String(s) => ";" == s,
                    pp::Token::Begin(_) => true,
                    pp::Token::End => true,
                    _ => false,
                };
                if twice {
                    self.hardbreak();
                }
                self.hardbreak();
            }
        }
        if let Some(cmnts) = self.comments() {
            cmnts.current += 1;
        }
    }

    fn next_comment(&mut self) -> Option<Comment> {
        self.comments().as_mut().and_then(|c| c.next())
    }

    fn print_literal(&mut self, lit: &ast::Lit) {
        self.maybe_print_comment(lit.span.lo());
        self.word(lit.token.to_string())
    }

    fn print_string(&mut self, st: &str, style: ast::StrStyle) {
        let st = match style {
            ast::StrStyle::Cooked => (format!("\"{}\"", st.escape_debug())),
            ast::StrStyle::Raw(n) => {
                format!("r{delim}\"{string}\"{delim}", delim = "#".repeat(n as usize), string = st)
            }
        };
        self.word(st)
    }

    fn print_symbol(&mut self, sym: Symbol, style: ast::StrStyle) {
        self.print_string(&sym.as_str(), style);
    }

    fn print_inner_attributes(&mut self, attrs: &[ast::Attribute]) {
        self.print_either_attributes(attrs, ast::AttrStyle::Inner, false, true)
    }

    fn print_inner_attributes_no_trailing_hardbreak(&mut self, attrs: &[ast::Attribute]) {
        self.print_either_attributes(attrs, ast::AttrStyle::Inner, false, false)
    }

    fn print_outer_attributes(&mut self, attrs: &[ast::Attribute]) {
        self.print_either_attributes(attrs, ast::AttrStyle::Outer, false, true)
    }

    fn print_inner_attributes_inline(&mut self, attrs: &[ast::Attribute]) {
        self.print_either_attributes(attrs, ast::AttrStyle::Inner, true, true)
    }

    fn print_outer_attributes_inline(&mut self, attrs: &[ast::Attribute]) {
        self.print_either_attributes(attrs, ast::AttrStyle::Outer, true, true)
    }

    fn print_either_attributes(
        &mut self,
        attrs: &[ast::Attribute],
        kind: ast::AttrStyle,
        is_inline: bool,
        trailing_hardbreak: bool,
    ) {
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

    fn print_attribute(&mut self, attr: &ast::Attribute) {
        self.print_attribute_inline(attr, false)
    }

    fn print_attribute_inline(&mut self, attr: &ast::Attribute, is_inline: bool) {
        if !is_inline {
            self.hardbreak_if_not_bol();
        }
        self.maybe_print_comment(attr.span.lo());
        match attr.kind {
            ast::AttrKind::Normal(ref item, _) => {
                match attr.style {
                    ast::AttrStyle::Inner => self.word("#!["),
                    ast::AttrStyle::Outer => self.word("#["),
                }
                self.print_attr_item(&item, attr.span);
                self.word("]");
            }
            ast::AttrKind::DocComment(comment_kind, data) => {
                self.word(doc_comment_to_string(comment_kind, attr.style, data));
                self.hardbreak()
            }
        }
    }

    fn print_attr_item(&mut self, item: &ast::AttrItem, span: Span) {
        self.ibox(0);
        match &item.args {
            MacArgs::Delimited(_, delim, tokens) => self.print_mac_common(
                Some(MacHeader::Path(&item.path)),
                false,
                None,
                delim.to_token(),
                tokens,
                true,
                span,
            ),
            MacArgs::Empty | MacArgs::Eq(..) => {
                self.print_path(&item.path, false, 0);
                if let MacArgs::Eq(_, tokens) = &item.args {
                    self.space();
                    self.word_space("=");
                    self.print_tts(tokens, true);
                }
            }
        }
        self.end();
    }

    fn print_meta_list_item(&mut self, item: &ast::NestedMetaItem) {
        match item {
            ast::NestedMetaItem::MetaItem(ref mi) => self.print_meta_item(mi),
            ast::NestedMetaItem::Literal(ref lit) => self.print_literal(lit),
        }
    }

    fn print_meta_item(&mut self, item: &ast::MetaItem) {
        self.ibox(INDENT_UNIT);
        match item.kind {
            ast::MetaItemKind::Word => self.print_path(&item.path, false, 0),
            ast::MetaItemKind::NameValue(ref value) => {
                self.print_path(&item.path, false, 0);
                self.space();
                self.word_space("=");
                self.print_literal(value);
            }
            ast::MetaItemKind::List(ref items) => {
                self.print_path(&item.path, false, 0);
                self.popen();
                self.commasep(Consistent, &items[..], |s, i| s.print_meta_list_item(i));
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
    fn print_tt(&mut self, tt: &TokenTree, convert_dollar_crate: bool) {
        match tt {
            TokenTree::Token(token) => {
                let token_str = self.token_to_string_ext(&token, convert_dollar_crate);
                self.word(token_str);
                if let token::DocComment(..) = token.kind {
                    self.hardbreak()
                }
            }
            TokenTree::Delimited(dspan, delim, tts) => {
                self.print_mac_common(
                    None,
                    false,
                    None,
                    *delim,
                    tts,
                    convert_dollar_crate,
                    dspan.entire(),
                );
            }
        }
    }

    fn print_tts(&mut self, tts: &TokenStream, convert_dollar_crate: bool) {
        let mut iter = tts.trees().peekable();
        while let Some(tt) = iter.next() {
            self.print_tt(&tt, convert_dollar_crate);
            if let Some(next) = iter.peek() {
                if tt_prepend_space(next, &tt) {
                    self.space();
                }
            }
        }
    }

    fn print_mac_common(
        &mut self,
        header: Option<MacHeader<'_>>,
        has_bang: bool,
        ident: Option<Ident>,
        delim: DelimToken,
        tts: &TokenStream,
        convert_dollar_crate: bool,
        span: Span,
    ) {
        if delim == DelimToken::Brace {
            self.cbox(INDENT_UNIT);
        }
        match header {
            Some(MacHeader::Path(path)) => self.print_path(path, false, 0),
            Some(MacHeader::Keyword(kw)) => self.word(kw),
            None => {}
        }
        if has_bang {
            self.word("!");
        }
        if let Some(ident) = ident {
            self.nbsp();
            self.print_ident(ident);
        }
        match delim {
            DelimToken::Brace => {
                if header.is_some() || has_bang || ident.is_some() {
                    self.nbsp();
                }
                self.word("{");
                if !tts.is_empty() {
                    self.space();
                }
            }
            _ => {
                let token_str = self.token_kind_to_string(&token::OpenDelim(delim));
                self.word(token_str)
            }
        }
        self.ibox(0);
        self.print_tts(tts, convert_dollar_crate);
        self.end();
        match delim {
            DelimToken::Brace => self.bclose(span),
            _ => {
                let token_str = self.token_kind_to_string(&token::CloseDelim(delim));
                self.word(token_str)
            }
        }
    }

    fn print_path(&mut self, path: &ast::Path, colons_before_params: bool, depth: usize) {
        self.maybe_print_comment(path.span.lo());

        for (i, segment) in path.segments[..path.segments.len() - depth].iter().enumerate() {
            if i > 0 {
                self.word("::")
            }
            self.print_path_segment(segment, colons_before_params);
        }
    }

    fn print_path_segment(&mut self, segment: &ast::PathSegment, colons_before_params: bool) {
        if segment.ident.name != kw::PathRoot {
            self.print_ident(segment.ident);
            if let Some(ref args) = segment.args {
                self.print_generic_args(args, colons_before_params);
            }
        }
    }

    fn head<S: Into<Cow<'static, str>>>(&mut self, w: S) {
        let w = w.into();
        // Outer-box is consistent.
        self.cbox(INDENT_UNIT);
        // Head-box is inconsistent.
        self.ibox(w.len() + 1);
        // Keyword that starts the head.
        if !w.is_empty() {
            self.word_nbsp(w);
        }
    }

    fn bopen(&mut self) {
        self.word("{");
        self.end(); // Close the head-box.
    }

    fn bclose_maybe_open(&mut self, span: rustc_span::Span, close_box: bool) {
        self.maybe_print_comment(span.hi());
        self.break_offset_if_not_bol(1, -(INDENT_UNIT as isize));
        self.word("}");
        if close_box {
            self.end(); // Close the outer-box.
        }
    }

    fn bclose(&mut self, span: rustc_span::Span) {
        self.bclose_maybe_open(span, true)
    }

    fn break_offset_if_not_bol(&mut self, n: usize, off: isize) {
        if !self.is_beginning_of_line() {
            self.break_offset(n, off)
        } else if off != 0 && self.last_token().is_hardbreak_tok() {
            // We do something pretty sketchy here: tuck the nonzero
            // offset-adjustment we were going to deposit along with the
            // break into the previous hardbreak.
            self.replace_last_token(pp::Printer::hardbreak_tok_offset(off));
        }
    }

    fn nonterminal_to_string(&self, nt: &Nonterminal) -> String {
        match *nt {
            token::NtExpr(ref e) => self.expr_to_string(e),
            token::NtMeta(ref e) => self.attr_item_to_string(e),
            token::NtTy(ref e) => self.ty_to_string(e),
            token::NtPath(ref e) => self.path_to_string(e),
            token::NtItem(ref e) => self.item_to_string(e),
            token::NtBlock(ref e) => self.block_to_string(e),
            token::NtStmt(ref e) => self.stmt_to_string(e),
            token::NtPat(ref e) => self.pat_to_string(e),
            token::NtIdent(e, is_raw) => IdentPrinter::for_ast_ident(e, is_raw).to_string(),
            token::NtLifetime(e) => e.to_string(),
            token::NtLiteral(ref e) => self.expr_to_string(e),
            token::NtTT(ref tree) => self.tt_to_string(tree),
            token::NtVis(ref e) => self.vis_to_string(e),
        }
    }

    /// Print the token kind precisely, without converting `$crate` into its respective crate name.
    fn token_kind_to_string(&self, tok: &TokenKind) -> String {
        self.token_kind_to_string_ext(tok, None)
    }

    fn token_kind_to_string_ext(
        &self,
        tok: &TokenKind,
        convert_dollar_crate: Option<Span>,
    ) -> String {
        match *tok {
            token::Eq => "=".to_string(),
            token::Lt => "<".to_string(),
            token::Le => "<=".to_string(),
            token::EqEq => "==".to_string(),
            token::Ne => "!=".to_string(),
            token::Ge => ">=".to_string(),
            token::Gt => ">".to_string(),
            token::Not => "!".to_string(),
            token::Tilde => "~".to_string(),
            token::OrOr => "||".to_string(),
            token::AndAnd => "&&".to_string(),
            token::BinOp(op) => binop_to_string(op).to_string(),
            token::BinOpEq(op) => format!("{}=", binop_to_string(op)),

            /* Structural symbols */
            token::At => "@".to_string(),
            token::Dot => ".".to_string(),
            token::DotDot => "..".to_string(),
            token::DotDotDot => "...".to_string(),
            token::DotDotEq => "..=".to_string(),
            token::Comma => ",".to_string(),
            token::Semi => ";".to_string(),
            token::Colon => ":".to_string(),
            token::ModSep => "::".to_string(),
            token::RArrow => "->".to_string(),
            token::LArrow => "<-".to_string(),
            token::FatArrow => "=>".to_string(),
            token::OpenDelim(token::Paren) => "(".to_string(),
            token::CloseDelim(token::Paren) => ")".to_string(),
            token::OpenDelim(token::Bracket) => "[".to_string(),
            token::CloseDelim(token::Bracket) => "]".to_string(),
            token::OpenDelim(token::Brace) => "{".to_string(),
            token::CloseDelim(token::Brace) => "}".to_string(),
            token::OpenDelim(token::NoDelim) | token::CloseDelim(token::NoDelim) => "".to_string(),
            token::Pound => "#".to_string(),
            token::Dollar => "$".to_string(),
            token::Question => "?".to_string(),
            token::SingleQuote => "'".to_string(),

            /* Literals */
            token::Literal(lit) => literal_to_string(lit),

            /* Name components */
            token::Ident(s, is_raw) => {
                IdentPrinter::new(s, is_raw, convert_dollar_crate).to_string()
            }
            token::Lifetime(s) => s.to_string(),

            /* Other */
            token::DocComment(comment_kind, attr_style, data) => {
                doc_comment_to_string(comment_kind, attr_style, data)
            }
            token::Eof => "<eof>".to_string(),

            token::Interpolated(ref nt) => self.nonterminal_to_string(nt),
        }
    }

    /// Print the token precisely, without converting `$crate` into its respective crate name.
    fn token_to_string(&self, token: &Token) -> String {
        self.token_to_string_ext(token, false)
    }

    fn token_to_string_ext(&self, token: &Token, convert_dollar_crate: bool) -> String {
        let convert_dollar_crate = convert_dollar_crate.then_some(token.span);
        self.token_kind_to_string_ext(&token.kind, convert_dollar_crate)
    }

    fn ty_to_string(&self, ty: &ast::Ty) -> String {
        self.to_string(|s| s.print_type(ty))
    }

    fn bounds_to_string(&self, bounds: &[ast::GenericBound]) -> String {
        self.to_string(|s| s.print_type_bounds("", bounds))
    }

    fn pat_to_string(&self, pat: &ast::Pat) -> String {
        self.to_string(|s| s.print_pat(pat))
    }

    fn expr_to_string(&self, e: &ast::Expr) -> String {
        self.to_string(|s| s.print_expr(e))
    }

    fn tt_to_string(&self, tt: &TokenTree) -> String {
        self.to_string(|s| s.print_tt(tt, false))
    }

    fn tts_to_string(&self, tokens: &TokenStream) -> String {
        self.to_string(|s| s.print_tts(tokens, false))
    }

    fn stmt_to_string(&self, stmt: &ast::Stmt) -> String {
        self.to_string(|s| s.print_stmt(stmt))
    }

    fn item_to_string(&self, i: &ast::Item) -> String {
        self.to_string(|s| s.print_item(i))
    }

    fn generic_params_to_string(&self, generic_params: &[ast::GenericParam]) -> String {
        self.to_string(|s| s.print_generic_params(generic_params))
    }

    fn path_to_string(&self, p: &ast::Path) -> String {
        self.to_string(|s| s.print_path(p, false, 0))
    }

    fn path_segment_to_string(&self, p: &ast::PathSegment) -> String {
        self.to_string(|s| s.print_path_segment(p, false))
    }

    fn vis_to_string(&self, v: &ast::Visibility) -> String {
        self.to_string(|s| s.print_visibility(v))
    }

    fn block_to_string(&self, blk: &ast::Block) -> String {
        self.to_string(|s| {
            // Containing cbox, will be closed by `print_block` at `}`.
            s.cbox(INDENT_UNIT);
            // Head-ibox, will be closed by `print_block` after `{`.
            s.ibox(0);
            s.print_block(blk)
        })
    }

    fn meta_list_item_to_string(&self, li: &ast::NestedMetaItem) -> String {
        self.to_string(|s| s.print_meta_list_item(li))
    }

    fn attr_item_to_string(&self, ai: &ast::AttrItem) -> String {
        self.to_string(|s| s.print_attr_item(ai, ai.path.span))
    }

    fn attribute_to_string(&self, attr: &ast::Attribute) -> String {
        self.to_string(|s| s.print_attribute(attr))
    }

    fn param_to_string(&self, arg: &ast::Param) -> String {
        self.to_string(|s| s.print_param(arg, false))
    }

    fn to_string(&self, f: impl FnOnce(&mut State<'_>)) -> String {
        let mut printer = State::new();
        printer.insert_extra_parens = self.insert_extra_parens();
        f(&mut printer);
        printer.s.eof()
    }
}

impl<'a> PrintState<'a> for State<'a> {
    fn insert_extra_parens(&self) -> bool {
        self.insert_extra_parens
    }
    fn comments(&mut self) -> &mut Option<Comments<'a>> {
        &mut self.comments
    }

    fn print_ident(&mut self, ident: Ident) {
        self.s.word(IdentPrinter::for_ast_ident(ident, ident.is_raw_guess()).to_string());
        self.ann.post(self, AnnNode::Ident(&ident))
    }

    fn print_generic_args(&mut self, args: &ast::GenericArgs, colons_before_params: bool) {
        if colons_before_params {
            self.s.word("::")
        }

        match *args {
            ast::GenericArgs::AngleBracketed(ref data) => {
                self.s.word("<");
                self.commasep(Inconsistent, &data.args, |s, arg| match arg {
                    ast::AngleBracketedArg::Arg(a) => s.print_generic_arg(a),
                    ast::AngleBracketedArg::Constraint(c) => s.print_assoc_constraint(c),
                });
                self.s.word(">")
            }

            ast::GenericArgs::Parenthesized(ref data) => {
                self.s.word("(");
                self.commasep(Inconsistent, &data.inputs, |s, ty| s.print_type(ty));
                self.s.word(")");
                self.print_fn_ret_ty(&data.output);
            }
        }
    }
}

impl<'a> State<'a> {
    pub fn new() -> State<'a> {
        State {
            s: pp::mk_printer(),
            comments: None,
            ann: &NoAnn,
            is_expanded: false,
            insert_extra_parens: true,
        }
    }

    pub(super) fn without_insert_extra_parens() -> State<'a> {
        State { insert_extra_parens: false, ..State::new() }
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

    crate fn commasep_cmnt<T, F, G>(&mut self, b: Breaks, elts: &[T], mut op: F, mut get_span: G)
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

    crate fn commasep_exprs(&mut self, b: Breaks, exprs: &[P<ast::Expr>]) {
        self.commasep_cmnt(b, exprs, |s, e| s.print_expr(e), |e| e.span)
    }

    pub fn print_mod(&mut self, _mod: &ast::Mod, attrs: &[ast::Attribute]) {
        self.print_inner_attributes(attrs);
        for item in &_mod.items {
            self.print_item(item);
        }
    }

    crate fn print_foreign_mod(&mut self, nmod: &ast::ForeignMod, attrs: &[ast::Attribute]) {
        self.print_inner_attributes(attrs);
        for item in &nmod.items {
            self.print_foreign_item(item);
        }
    }

    pub fn print_opt_lifetime(&mut self, lifetime: &Option<ast::Lifetime>) {
        if let Some(lt) = *lifetime {
            self.print_lifetime(lt);
            self.nbsp();
        }
    }

    pub fn print_assoc_constraint(&mut self, constraint: &ast::AssocTyConstraint) {
        self.print_ident(constraint.ident);
        self.s.space();
        match &constraint.kind {
            ast::AssocTyConstraintKind::Equality { ty } => {
                self.word_space("=");
                self.print_type(ty);
            }
            ast::AssocTyConstraintKind::Bound { bounds } => {
                self.print_type_bounds(":", &*bounds);
            }
        }
    }

    pub fn print_generic_arg(&mut self, generic_arg: &GenericArg) {
        match generic_arg {
            GenericArg::Lifetime(lt) => self.print_lifetime(*lt),
            GenericArg::Type(ty) => self.print_type(ty),
            GenericArg::Const(ct) => self.print_expr(&ct.value),
        }
    }

    pub fn print_type(&mut self, ty: &ast::Ty) {
        self.maybe_print_comment(ty.span.lo());
        self.ibox(0);
        match ty.kind {
            ast::TyKind::Slice(ref ty) => {
                self.s.word("[");
                self.print_type(ty);
                self.s.word("]");
            }
            ast::TyKind::Ptr(ref mt) => {
                self.s.word("*");
                self.print_mt(mt, true);
            }
            ast::TyKind::Rptr(ref lifetime, ref mt) => {
                self.s.word("&");
                self.print_opt_lifetime(lifetime);
                self.print_mt(mt, false);
            }
            ast::TyKind::Never => {
                self.s.word("!");
            }
            ast::TyKind::Tup(ref elts) => {
                self.popen();
                self.commasep(Inconsistent, &elts[..], |s, ty| s.print_type(ty));
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
                self.print_ty_fn(f.ext, f.unsafety, &f.decl, None, &f.generic_params);
            }
            ast::TyKind::Path(None, ref path) => {
                self.print_path(path, false, 0);
            }
            ast::TyKind::Path(Some(ref qself), ref path) => self.print_qpath(path, qself, false),
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
            ast::TyKind::MacCall(ref m) => {
                self.print_mac(m);
            }
            ast::TyKind::CVarArgs => {
                self.s.word("...");
            }
        }
        self.end();
    }

    crate fn print_foreign_item(&mut self, item: &ast::ForeignItem) {
        let ast::Item { id, span, ident, ref attrs, ref kind, ref vis, tokens: _ } = *item;
        self.ann.pre(self, AnnNode::SubItem(id));
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(span.lo());
        self.print_outer_attributes(attrs);
        match kind {
            ast::ForeignItemKind::Fn(def, sig, gen, body) => {
                self.print_fn_full(sig, ident, gen, vis, *def, body.as_deref(), attrs);
            }
            ast::ForeignItemKind::Static(ty, mutbl, body) => {
                let def = ast::Defaultness::Final;
                self.print_item_const(ident, Some(*mutbl), ty, body.as_deref(), vis, def);
            }
            ast::ForeignItemKind::TyAlias(def, generics, bounds, ty) => {
                self.print_associated_type(ident, generics, bounds, ty.as_deref(), vis, *def);
            }
            ast::ForeignItemKind::MacCall(m) => {
                self.print_mac(m);
                if m.args.need_semicolon() {
                    self.s.word(";");
                }
            }
        }
        self.ann.post(self, AnnNode::SubItem(id))
    }

    fn print_item_const(
        &mut self,
        ident: Ident,
        mutbl: Option<ast::Mutability>,
        ty: &ast::Ty,
        body: Option<&ast::Expr>,
        vis: &ast::Visibility,
        defaultness: ast::Defaultness,
    ) {
        self.head("");
        self.print_visibility(vis);
        self.print_defaultness(defaultness);
        let leading = match mutbl {
            None => "const",
            Some(ast::Mutability::Not) => "static",
            Some(ast::Mutability::Mut) => "static mut",
        };
        self.word_space(leading);
        self.print_ident(ident);
        self.word_space(":");
        self.print_type(ty);
        self.s.space();
        self.end(); // end the head-ibox
        if let Some(body) = body {
            self.word_space("=");
            self.print_expr(body);
        }
        self.s.word(";");
        self.end(); // end the outer cbox
    }

    fn print_associated_type(
        &mut self,
        ident: Ident,
        generics: &ast::Generics,
        bounds: &ast::GenericBounds,
        ty: Option<&ast::Ty>,
        vis: &ast::Visibility,
        defaultness: ast::Defaultness,
    ) {
        self.head("");
        self.print_visibility(vis);
        self.print_defaultness(defaultness);
        self.word_space("type");
        self.print_ident(ident);
        self.print_generic_params(&generics.params);
        self.print_type_bounds(":", bounds);
        self.print_where_clause(&generics.where_clause);
        if let Some(ty) = ty {
            self.s.space();
            self.word_space("=");
            self.print_type(ty);
        }
        self.s.word(";");
        self.end(); // end inner head-block
        self.end(); // end outer head-block
    }

    /// Pretty-prints an item.
    crate fn print_item(&mut self, item: &ast::Item) {
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(item.span.lo());
        self.print_outer_attributes(&item.attrs);
        self.ann.pre(self, AnnNode::Item(item));
        match item.kind {
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
            ast::ItemKind::Static(ref ty, mutbl, ref body) => {
                let def = ast::Defaultness::Final;
                self.print_item_const(item.ident, Some(mutbl), ty, body.as_deref(), &item.vis, def);
            }
            ast::ItemKind::Const(def, ref ty, ref body) => {
                self.print_item_const(item.ident, None, ty, body.as_deref(), &item.vis, def);
            }
            ast::ItemKind::Fn(def, ref sig, ref gen, ref body) => {
                let body = body.as_deref();
                self.print_fn_full(sig, item.ident, gen, &item.vis, def, body, &item.attrs);
            }
            ast::ItemKind::Mod(ref _mod) => {
                self.head(self.to_string(|s| {
                    s.print_visibility(&item.vis);
                    s.print_unsafety(_mod.unsafety);
                    s.word("mod");
                }));
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
                self.head(self.to_string(|s| {
                    s.print_unsafety(nmod.unsafety);
                    s.word("extern");
                }));
                if let Some(abi) = nmod.abi {
                    self.print_literal(&abi.as_lit());
                    self.nbsp();
                }
                self.bopen();
                self.print_foreign_mod(nmod, &item.attrs);
                self.bclose(item.span);
            }
            ast::ItemKind::GlobalAsm(ref ga) => {
                self.head(visibility_qualified(&item.vis, "global_asm!"));
                self.s.word(ga.asm.to_string());
                self.end();
            }
            ast::ItemKind::TyAlias(def, ref generics, ref bounds, ref ty) => {
                let ty = ty.as_deref();
                self.print_associated_type(item.ident, generics, bounds, ty, &item.vis, def);
            }
            ast::ItemKind::Enum(ref enum_definition, ref params) => {
                self.print_enum_def(enum_definition, params, item.ident, item.span, &item.vis);
            }
            ast::ItemKind::Struct(ref struct_def, ref generics) => {
                self.head(visibility_qualified(&item.vis, "struct"));
                self.print_struct(struct_def, generics, item.ident, item.span, true);
            }
            ast::ItemKind::Union(ref struct_def, ref generics) => {
                self.head(visibility_qualified(&item.vis, "union"));
                self.print_struct(struct_def, generics, item.ident, item.span, true);
            }
            ast::ItemKind::Impl {
                unsafety,
                polarity,
                defaultness,
                constness,
                ref generics,
                ref of_trait,
                ref self_ty,
                ref items,
            } => {
                self.head("");
                self.print_visibility(&item.vis);
                self.print_defaultness(defaultness);
                self.print_unsafety(unsafety);
                self.word_nbsp("impl");
                self.print_constness(constness);

                if !generics.params.is_empty() {
                    self.print_generic_params(&generics.params);
                    self.s.space();
                }

                if let ast::ImplPolarity::Negative(_) = polarity {
                    self.s.word("!");
                }

                if let Some(ref t) = *of_trait {
                    self.print_trait_ref(t);
                    self.s.space();
                    self.word_space("for");
                }

                self.print_type(self_ty);
                self.print_where_clause(&generics.where_clause);

                self.s.space();
                self.bopen();
                self.print_inner_attributes(&item.attrs);
                for impl_item in items {
                    self.print_assoc_item(impl_item);
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
                self.print_inner_attributes(&item.attrs);
                for trait_item in trait_items {
                    self.print_assoc_item(trait_item);
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
            ast::ItemKind::MacCall(ref mac) => {
                self.print_mac(mac);
                if mac.args.need_semicolon() {
                    self.s.word(";");
                }
            }
            ast::ItemKind::MacroDef(ref macro_def) => {
                let (kw, has_bang) = if macro_def.macro_rules {
                    ("macro_rules", true)
                } else {
                    self.print_visibility(&item.vis);
                    ("macro", false)
                };
                self.print_mac_common(
                    Some(MacHeader::Keyword(kw)),
                    has_bang,
                    Some(item.ident),
                    macro_def.body.delim(),
                    &macro_def.body.inner_tokens(),
                    true,
                    item.span,
                );
            }
        }
        self.ann.post(self, AnnNode::Item(item))
    }

    fn print_trait_ref(&mut self, t: &ast::TraitRef) {
        self.print_path(&t.path, false, 0)
    }

    fn print_formal_generic_params(&mut self, generic_params: &[ast::GenericParam]) {
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

    crate fn print_enum_def(
        &mut self,
        enum_definition: &ast::EnumDef,
        generics: &ast::Generics,
        ident: Ident,
        span: rustc_span::Span,
        visibility: &ast::Visibility,
    ) {
        self.head(visibility_qualified(visibility, "enum"));
        self.print_ident(ident);
        self.print_generic_params(&generics.params);
        self.print_where_clause(&generics.where_clause);
        self.s.space();
        self.print_variants(&enum_definition.variants, span)
    }

    crate fn print_variants(&mut self, variants: &[ast::Variant], span: rustc_span::Span) {
        self.bopen();
        for v in variants {
            self.space_if_not_bol();
            self.maybe_print_comment(v.span.lo());
            self.print_outer_attributes(&v.attrs);
            self.ibox(INDENT_UNIT);
            self.print_variant(v);
            self.s.word(",");
            self.end();
            self.maybe_print_trailing_comment(v.span, None);
        }
        self.bclose(span)
    }

    crate fn print_visibility(&mut self, vis: &ast::Visibility) {
        match vis.kind {
            ast::VisibilityKind::Public => self.word_nbsp("pub"),
            ast::VisibilityKind::Crate(sugar) => match sugar {
                ast::CrateSugar::PubCrate => self.word_nbsp("pub(crate)"),
                ast::CrateSugar::JustCrate => self.word_nbsp("crate"),
            },
            ast::VisibilityKind::Restricted { ref path, .. } => {
                let path = self.to_string(|s| s.print_path(path, false, 0));
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
        if let ast::Defaultness::Default(_) = defaultness {
            self.word_nbsp("default");
        }
    }

    crate fn print_struct(
        &mut self,
        struct_def: &ast::VariantData,
        generics: &ast::Generics,
        ident: Ident,
        span: rustc_span::Span,
        print_finalizer: bool,
    ) {
        self.print_ident(ident);
        self.print_generic_params(&generics.params);
        match struct_def {
            ast::VariantData::Tuple(..) | ast::VariantData::Unit(..) => {
                if let ast::VariantData::Tuple(..) = struct_def {
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
                self.end(); // Close the outer-box.
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
        self.print_visibility(&v.vis);
        let generics = ast::Generics::default();
        self.print_struct(&v.data, &generics, v.ident, v.span, false);
        if let Some(ref d) = v.disr_expr {
            self.s.space();
            self.word_space("=");
            self.print_expr(&d.value)
        }
    }

    crate fn print_assoc_item(&mut self, item: &ast::AssocItem) {
        let ast::Item { id, span, ident, ref attrs, ref kind, ref vis, tokens: _ } = *item;
        self.ann.pre(self, AnnNode::SubItem(id));
        self.hardbreak_if_not_bol();
        self.maybe_print_comment(span.lo());
        self.print_outer_attributes(attrs);
        match kind {
            ast::AssocItemKind::Fn(def, sig, gen, body) => {
                self.print_fn_full(sig, ident, gen, vis, *def, body.as_deref(), attrs);
            }
            ast::AssocItemKind::Const(def, ty, body) => {
                self.print_item_const(ident, None, ty, body.as_deref(), vis, *def);
            }
            ast::AssocItemKind::TyAlias(def, generics, bounds, ty) => {
                self.print_associated_type(ident, generics, bounds, ty.as_deref(), vis, *def);
            }
            ast::AssocItemKind::MacCall(m) => {
                self.print_mac(m);
                if m.args.need_semicolon() {
                    self.s.word(";");
                }
            }
        }
        self.ann.post(self, AnnNode::SubItem(id))
    }

    crate fn print_stmt(&mut self, st: &ast::Stmt) {
        self.maybe_print_comment(st.span.lo());
        match st.kind {
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
                if classify::expr_requires_semi_to_be_stmt(expr) {
                    self.s.word(";");
                }
            }
            ast::StmtKind::Semi(ref expr) => {
                self.space_if_not_bol();
                self.print_expr_outer_attr_style(expr, false);
                self.s.word(";");
            }
            ast::StmtKind::Empty => {
                self.space_if_not_bol();
                self.s.word(";");
            }
            ast::StmtKind::MacCall(ref mac) => {
                self.space_if_not_bol();
                self.print_outer_attributes(&mac.attrs);
                self.print_mac(&mac.mac);
                if mac.style == ast::MacStmtStyle::Semicolon {
                    self.s.word(";");
                }
            }
        }
        self.maybe_print_trailing_comment(st.span, None)
    }

    crate fn print_block(&mut self, blk: &ast::Block) {
        self.print_block_with_attrs(blk, &[])
    }

    crate fn print_block_unclosed_indent(&mut self, blk: &ast::Block) {
        self.print_block_maybe_unclosed(blk, &[], false)
    }

    crate fn print_block_with_attrs(&mut self, blk: &ast::Block, attrs: &[ast::Attribute]) {
        self.print_block_maybe_unclosed(blk, attrs, true)
    }

    crate fn print_block_maybe_unclosed(
        &mut self,
        blk: &ast::Block,
        attrs: &[ast::Attribute],
        close_box: bool,
    ) {
        match blk.rules {
            BlockCheckMode::Unsafe(..) => self.word_space("unsafe"),
            BlockCheckMode::Default => (),
        }
        self.maybe_print_comment(blk.span.lo());
        self.ann.pre(self, AnnNode::Block(blk));
        self.bopen();

        self.print_inner_attributes(attrs);

        for (i, st) in blk.stmts.iter().enumerate() {
            match st.kind {
                ast::StmtKind::Expr(ref expr) if i == blk.stmts.len() - 1 => {
                    self.maybe_print_comment(st.span.lo());
                    self.space_if_not_bol();
                    self.print_expr_outer_attr_style(expr, false);
                    self.maybe_print_trailing_comment(expr.span, Some(blk.span.hi()));
                }
                _ => self.print_stmt(st),
            }
        }

        self.bclose_maybe_open(blk.span, close_box);
        self.ann.post(self, AnnNode::Block(blk))
    }

    /// Print a `let pat = scrutinee` expression.
    crate fn print_let(&mut self, pat: &ast::Pat, scrutinee: &ast::Expr) {
        self.s.word("let ");

        self.print_pat(pat);
        self.s.space();

        self.word_space("=");
        self.print_expr_cond_paren(
            scrutinee,
            Self::cond_needs_par(scrutinee)
                || parser::needs_par_as_let_scrutinee(scrutinee.precedence().order()),
        )
    }

    fn print_else(&mut self, els: Option<&ast::Expr>) {
        if let Some(_else) = els {
            match _else.kind {
                // Another `else if` block.
                ast::ExprKind::If(ref i, ref then, ref e) => {
                    self.cbox(INDENT_UNIT - 1);
                    self.ibox(0);
                    self.s.word(" else if ");
                    self.print_expr_as_cond(i);
                    self.s.space();
                    self.print_block(then);
                    self.print_else(e.as_deref())
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
    }

    crate fn print_if(&mut self, test: &ast::Expr, blk: &ast::Block, elseopt: Option<&ast::Expr>) {
        self.head("if");

        self.print_expr_as_cond(test);
        self.s.space();

        self.print_block(blk);
        self.print_else(elseopt)
    }

    crate fn print_mac(&mut self, m: &ast::MacCall) {
        self.print_mac_common(
            Some(MacHeader::Path(&m.path)),
            true,
            None,
            m.args.delim(),
            &m.args.inner_tokens(),
            true,
            m.span(),
        );
    }

    fn print_call_post(&mut self, args: &[P<ast::Expr>]) {
        self.popen();
        self.commasep_exprs(Inconsistent, args);
        self.pclose()
    }

    crate fn print_expr_maybe_paren(&mut self, expr: &ast::Expr, prec: i8) {
        self.print_expr_cond_paren(expr, expr.precedence().order() < prec)
    }

    /// Prints an expr using syntax that's acceptable in a condition position, such as the `cond` in
    /// `if cond { ... }`.
    crate fn print_expr_as_cond(&mut self, expr: &ast::Expr) {
        self.print_expr_cond_paren(expr, Self::cond_needs_par(expr))
    }

    /// Does `expr` need parenthesis when printed in a condition position?
    fn cond_needs_par(expr: &ast::Expr) -> bool {
        match expr.kind {
            // These cases need parens due to the parse error observed in #26461: `if return {}`
            // parses as the erroneous construct `if (return {})`, not `if (return) {}`.
            ast::ExprKind::Closure(..) | ast::ExprKind::Ret(..) | ast::ExprKind::Break(..) => true,

            _ => parser::contains_exterior_struct_lit(expr),
        }
    }

    /// Prints `expr` or `(expr)` when `needs_par` holds.
    fn print_expr_cond_paren(&mut self, expr: &ast::Expr, mut needs_par: bool) {
        needs_par &= self.insert_extra_parens;
        if needs_par {
            self.popen();
        }
        self.print_expr(expr);
        if needs_par {
            self.pclose();
        }
    }

    fn print_expr_vec(&mut self, exprs: &[P<ast::Expr>], attrs: &[ast::Attribute]) {
        self.ibox(INDENT_UNIT);
        self.s.word("[");
        self.print_inner_attributes_inline(attrs);
        self.commasep_exprs(Inconsistent, &exprs[..]);
        self.s.word("]");
        self.end();
    }

    fn print_expr_anon_const(&mut self, expr: &ast::AnonConst, attrs: &[ast::Attribute]) {
        self.ibox(INDENT_UNIT);
        self.s.word("const");
        self.print_inner_attributes_inline(attrs);
        self.print_expr(&expr.value);
        self.end();
    }

    fn print_expr_repeat(
        &mut self,
        element: &ast::Expr,
        count: &ast::AnonConst,
        attrs: &[ast::Attribute],
    ) {
        self.ibox(INDENT_UNIT);
        self.s.word("[");
        self.print_inner_attributes_inline(attrs);
        self.print_expr(element);
        self.word_space(";");
        self.print_expr(&count.value);
        self.s.word("]");
        self.end();
    }

    fn print_expr_struct(
        &mut self,
        path: &ast::Path,
        fields: &[ast::Field],
        rest: &ast::StructRest,
        attrs: &[ast::Attribute],
    ) {
        self.print_path(path, true, 0);
        self.s.word("{");
        self.print_inner_attributes_inline(attrs);
        self.commasep_cmnt(
            Consistent,
            &fields[..],
            |s, field| {
                s.print_outer_attributes(&field.attrs);
                s.ibox(INDENT_UNIT);
                if !field.is_shorthand {
                    s.print_ident(field.ident);
                    s.word_space(":");
                }
                s.print_expr(&field.expr);
                s.end();
            },
            |f| f.span,
        );
        match rest {
            ast::StructRest::Base(_) | ast::StructRest::Rest(_) => {
                self.ibox(INDENT_UNIT);
                if !fields.is_empty() {
                    self.s.word(",");
                    self.s.space();
                }
                self.s.word("..");
                if let ast::StructRest::Base(ref expr) = *rest {
                    self.print_expr(expr);
                }
                self.end();
            }
            ast::StructRest::None if !fields.is_empty() => self.s.word(","),
            _ => {}
        }
        self.s.word("}");
    }

    fn print_expr_tup(&mut self, exprs: &[P<ast::Expr>], attrs: &[ast::Attribute]) {
        self.popen();
        self.print_inner_attributes_inline(attrs);
        self.commasep_exprs(Inconsistent, &exprs[..]);
        if exprs.len() == 1 {
            self.s.word(",");
        }
        self.pclose()
    }

    fn print_expr_call(&mut self, func: &ast::Expr, args: &[P<ast::Expr>]) {
        let prec = match func.kind {
            ast::ExprKind::Field(..) => parser::PREC_FORCE_PAREN,
            _ => parser::PREC_POSTFIX,
        };

        self.print_expr_maybe_paren(func, prec);
        self.print_call_post(args)
    }

    fn print_expr_method_call(&mut self, segment: &ast::PathSegment, args: &[P<ast::Expr>]) {
        let base_args = &args[1..];
        self.print_expr_maybe_paren(&args[0], parser::PREC_POSTFIX);
        self.s.word(".");
        self.print_ident(segment.ident);
        if let Some(ref args) = segment.args {
            self.print_generic_args(args, true);
        }
        self.print_call_post(base_args)
    }

    fn print_expr_binary(&mut self, op: ast::BinOp, lhs: &ast::Expr, rhs: &ast::Expr) {
        let assoc_op = AssocOp::from_ast_binop(op.node);
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
            (&ast::ExprKind::Cast { .. }, ast::BinOpKind::Lt | ast::BinOpKind::Shl) => {
                parser::PREC_FORCE_PAREN
            }
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

    fn print_expr_unary(&mut self, op: ast::UnOp, expr: &ast::Expr) {
        self.s.word(ast::UnOp::to_string(op));
        self.print_expr_maybe_paren(expr, parser::PREC_PREFIX)
    }

    fn print_expr_addr_of(
        &mut self,
        kind: ast::BorrowKind,
        mutability: ast::Mutability,
        expr: &ast::Expr,
    ) {
        self.s.word("&");
        match kind {
            ast::BorrowKind::Ref => self.print_mutability(mutability, false),
            ast::BorrowKind::Raw => {
                self.word_nbsp("raw");
                self.print_mutability(mutability, true);
            }
        }
        self.print_expr_maybe_paren(expr, parser::PREC_PREFIX)
    }

    pub fn print_expr(&mut self, expr: &ast::Expr) {
        self.print_expr_outer_attr_style(expr, true)
    }

    fn print_expr_outer_attr_style(&mut self, expr: &ast::Expr, is_inline: bool) {
        self.maybe_print_comment(expr.span.lo());

        let attrs = &expr.attrs;
        if is_inline {
            self.print_outer_attributes_inline(attrs);
        } else {
            self.print_outer_attributes(attrs);
        }

        self.ibox(INDENT_UNIT);
        self.ann.pre(self, AnnNode::Expr(expr));
        match expr.kind {
            ast::ExprKind::Box(ref expr) => {
                self.word_space("box");
                self.print_expr_maybe_paren(expr, parser::PREC_PREFIX);
            }
            ast::ExprKind::Array(ref exprs) => {
                self.print_expr_vec(&exprs[..], attrs);
            }
            ast::ExprKind::ConstBlock(ref anon_const) => {
                self.print_expr_anon_const(anon_const, attrs);
            }
            ast::ExprKind::Repeat(ref element, ref count) => {
                self.print_expr_repeat(element, count, attrs);
            }
            ast::ExprKind::Struct(ref path, ref fields, ref rest) => {
                self.print_expr_struct(path, &fields[..], rest, attrs);
            }
            ast::ExprKind::Tup(ref exprs) => {
                self.print_expr_tup(&exprs[..], attrs);
            }
            ast::ExprKind::Call(ref func, ref args) => {
                self.print_expr_call(func, &args[..]);
            }
            ast::ExprKind::MethodCall(ref segment, ref args, _) => {
                self.print_expr_method_call(segment, &args[..]);
            }
            ast::ExprKind::Binary(op, ref lhs, ref rhs) => {
                self.print_expr_binary(op, lhs, rhs);
            }
            ast::ExprKind::Unary(op, ref expr) => {
                self.print_expr_unary(op, expr);
            }
            ast::ExprKind::AddrOf(k, m, ref expr) => {
                self.print_expr_addr_of(k, m, expr);
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
            ast::ExprKind::Let(ref pat, ref scrutinee) => {
                self.print_let(pat, scrutinee);
            }
            ast::ExprKind::If(ref test, ref blk, ref elseopt) => {
                self.print_if(test, blk, elseopt.as_deref())
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
                self.ibox(INDENT_UNIT);
                self.word_nbsp("match");
                self.print_expr_as_cond(expr);
                self.s.space();
                self.bopen();
                self.print_inner_attributes_no_trailing_hardbreak(attrs);
                for arm in arms {
                    self.print_arm(arm);
                }
                self.bclose(expr.span);
            }
            ast::ExprKind::Closure(
                capture_clause,
                asyncness,
                movability,
                ref decl,
                ref body,
                _,
            ) => {
                self.print_movability(movability);
                self.print_asyncness(asyncness);
                self.print_capture_clause(capture_clause);

                self.print_fn_params_and_ret(decl, true);
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
            ast::ExprKind::Await(ref expr) => {
                self.print_expr_maybe_paren(expr, parser::PREC_POSTFIX);
                self.s.word(".await");
            }
            ast::ExprKind::Assign(ref lhs, ref rhs, _) => {
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
            ast::ExprKind::Underscore => self.s.word("_"),
            ast::ExprKind::Path(None, ref path) => self.print_path(path, true, 0),
            ast::ExprKind::Path(Some(ref qself), ref path) => self.print_qpath(path, qself, true),
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
                enum AsmArg<'a> {
                    Template(String),
                    Operand(&'a InlineAsmOperand),
                    Options(InlineAsmOptions),
                }

                let mut args = vec![];
                args.push(AsmArg::Template(InlineAsmTemplatePiece::to_string(&a.template)));
                args.extend(a.operands.iter().map(|(o, _)| AsmArg::Operand(o)));
                if !a.options.is_empty() {
                    args.push(AsmArg::Options(a.options));
                }

                self.word("asm!");
                self.popen();
                self.commasep(Consistent, &args, |s, arg| match arg {
                    AsmArg::Template(template) => s.print_string(&template, ast::StrStyle::Cooked),
                    AsmArg::Operand(op) => {
                        let print_reg_or_class = |s: &mut Self, r: &InlineAsmRegOrRegClass| match r
                        {
                            InlineAsmRegOrRegClass::Reg(r) => {
                                s.print_symbol(*r, ast::StrStyle::Cooked)
                            }
                            InlineAsmRegOrRegClass::RegClass(r) => s.word(r.to_string()),
                        };
                        match op {
                            InlineAsmOperand::In { reg, expr } => {
                                s.word("in");
                                s.popen();
                                print_reg_or_class(s, reg);
                                s.pclose();
                                s.space();
                                s.print_expr(expr);
                            }
                            InlineAsmOperand::Out { reg, late, expr } => {
                                s.word(if *late { "lateout" } else { "out" });
                                s.popen();
                                print_reg_or_class(s, reg);
                                s.pclose();
                                s.space();
                                match expr {
                                    Some(expr) => s.print_expr(expr),
                                    None => s.word("_"),
                                }
                            }
                            InlineAsmOperand::InOut { reg, late, expr } => {
                                s.word(if *late { "inlateout" } else { "inout" });
                                s.popen();
                                print_reg_or_class(s, reg);
                                s.pclose();
                                s.space();
                                s.print_expr(expr);
                            }
                            InlineAsmOperand::SplitInOut { reg, late, in_expr, out_expr } => {
                                s.word(if *late { "inlateout" } else { "inout" });
                                s.popen();
                                print_reg_or_class(s, reg);
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
                            InlineAsmOperand::Const { expr } => {
                                s.word("const");
                                s.space();
                                s.print_expr(expr);
                            }
                            InlineAsmOperand::Sym { expr } => {
                                s.word("sym");
                                s.space();
                                s.print_expr(expr);
                            }
                        }
                    }
                    AsmArg::Options(opts) => {
                        s.word("options");
                        s.popen();
                        let mut options = vec![];
                        if opts.contains(InlineAsmOptions::PURE) {
                            options.push("pure");
                        }
                        if opts.contains(InlineAsmOptions::NOMEM) {
                            options.push("nomem");
                        }
                        if opts.contains(InlineAsmOptions::READONLY) {
                            options.push("readonly");
                        }
                        if opts.contains(InlineAsmOptions::PRESERVES_FLAGS) {
                            options.push("preserves_flags");
                        }
                        if opts.contains(InlineAsmOptions::NORETURN) {
                            options.push("noreturn");
                        }
                        if opts.contains(InlineAsmOptions::NOSTACK) {
                            options.push("nostack");
                        }
                        if opts.contains(InlineAsmOptions::ATT_SYNTAX) {
                            options.push("att_syntax");
                        }
                        s.commasep(Inconsistent, &options, |s, &opt| {
                            s.word(opt);
                        });
                        s.pclose();
                    }
                });
                self.pclose();
            }
            ast::ExprKind::LlvmInlineAsm(ref a) => {
                self.s.word("llvm_asm!");
                self.popen();
                self.print_symbol(a.asm, a.asm_str_style);
                self.word_space(":");

                self.commasep(Inconsistent, &a.outputs, |s, out| {
                    let constraint = out.constraint.as_str();
                    let mut ch = constraint.chars();
                    match ch.next() {
                        Some('=') if out.is_rw => {
                            s.print_string(&format!("+{}", ch.as_str()), ast::StrStyle::Cooked)
                        }
                        _ => s.print_string(&constraint, ast::StrStyle::Cooked),
                    }
                    s.popen();
                    s.print_expr(&out.expr);
                    s.pclose();
                });
                self.s.space();
                self.word_space(":");

                self.commasep(Inconsistent, &a.inputs, |s, &(co, ref o)| {
                    s.print_symbol(co, ast::StrStyle::Cooked);
                    s.popen();
                    s.print_expr(o);
                    s.pclose();
                });
                self.s.space();
                self.word_space(":");

                self.commasep(Inconsistent, &a.clobbers, |s, &co| {
                    s.print_symbol(co, ast::StrStyle::Cooked);
                });

                let mut options = vec![];
                if a.volatile {
                    options.push("volatile");
                }
                if a.alignstack {
                    options.push("alignstack");
                }
                if a.dialect == ast::LlvmAsmDialect::Intel {
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
            ast::ExprKind::MacCall(ref m) => self.print_mac(m),
            ast::ExprKind::Paren(ref e) => {
                self.popen();
                self.print_inner_attributes_inline(attrs);
                self.print_expr(e);
                self.pclose();
            }
            ast::ExprKind::Yield(ref e) => {
                self.s.word("yield");

                if let Some(ref expr) = *e {
                    self.s.space();
                    self.print_expr_maybe_paren(expr, parser::PREC_JUMP);
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

    pub fn print_usize(&mut self, i: usize) {
        self.s.word(i.to_string())
    }

    crate fn print_name(&mut self, name: Symbol) {
        self.s.word(name.to_string());
        self.ann.post(self, AnnNode::Name(&name))
    }

    fn print_qpath(&mut self, path: &ast::Path, qself: &ast::QSelf, colons_before_params: bool) {
        self.s.word("<");
        self.print_type(&qself.ty);
        if qself.position > 0 {
            self.s.space();
            self.word_space("as");
            let depth = path.segments.len() - qself.position;
            self.print_path(path, false, depth);
        }
        self.s.word(">");
        for item_segment in &path.segments[qself.position..] {
            self.s.word("::");
            self.print_ident(item_segment.ident);
            if let Some(ref args) = item_segment.args {
                self.print_generic_args(args, colons_before_params)
            }
        }
    }

    crate fn print_pat(&mut self, pat: &ast::Pat) {
        self.maybe_print_comment(pat.span.lo());
        self.ann.pre(self, AnnNode::Pat(pat));
        /* Pat isn't normalized, but the beauty of it
        is that it doesn't matter */
        match pat.kind {
            PatKind::Wild => self.s.word("_"),
            PatKind::Ident(binding_mode, ident, ref sub) => {
                match binding_mode {
                    ast::BindingMode::ByRef(mutbl) => {
                        self.word_nbsp("ref");
                        self.print_mutability(mutbl, false);
                    }
                    ast::BindingMode::ByValue(ast::Mutability::Not) => {}
                    ast::BindingMode::ByValue(ast::Mutability::Mut) => {
                        self.word_nbsp("mut");
                    }
                }
                self.print_ident(ident);
                if let Some(ref p) = *sub {
                    self.s.space();
                    self.s.word_space("@");
                    self.print_pat(p);
                }
            }
            PatKind::TupleStruct(ref path, ref elts) => {
                self.print_path(path, true, 0);
                self.popen();
                self.commasep(Inconsistent, &elts[..], |s, p| s.print_pat(p));
                self.pclose();
            }
            PatKind::Or(ref pats) => {
                self.strsep("|", true, Inconsistent, &pats[..], |s, p| s.print_pat(p));
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
                    Consistent,
                    &fields[..],
                    |s, f| {
                        s.cbox(INDENT_UNIT);
                        if !f.is_shorthand {
                            s.print_ident(f.ident);
                            s.word_nbsp(":");
                        }
                        s.print_pat(&f.pat);
                        s.end();
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
            PatKind::Tuple(ref elts) => {
                self.popen();
                self.commasep(Inconsistent, &elts[..], |s, p| s.print_pat(p));
                if elts.len() == 1 {
                    self.s.word(",");
                }
                self.pclose();
            }
            PatKind::Box(ref inner) => {
                self.s.word("box ");
                self.print_pat(inner);
            }
            PatKind::Ref(ref inner, mutbl) => {
                self.s.word("&");
                if mutbl == ast::Mutability::Mut {
                    self.s.word("mut ");
                }
                if let PatKind::Ident(ast::BindingMode::ByValue(ast::Mutability::Mut), ..) =
                    inner.kind
                {
                    self.popen();
                    self.print_pat(inner);
                    self.pclose();
                } else {
                    self.print_pat(inner);
                }
            }
            PatKind::Lit(ref e) => self.print_expr(&**e),
            PatKind::Range(ref begin, ref end, Spanned { node: ref end_kind, .. }) => {
                if let Some(e) = begin {
                    self.print_expr(e);
                    self.s.space();
                }
                match *end_kind {
                    RangeEnd::Included(RangeSyntax::DotDotDot) => self.s.word("..."),
                    RangeEnd::Included(RangeSyntax::DotDotEq) => self.s.word("..="),
                    RangeEnd::Excluded => self.s.word(".."),
                }
                if let Some(e) = end {
                    self.print_expr(e);
                }
            }
            PatKind::Slice(ref elts) => {
                self.s.word("[");
                self.commasep(Inconsistent, &elts[..], |s, p| s.print_pat(p));
                self.s.word("]");
            }
            PatKind::Rest => self.s.word(".."),
            PatKind::Paren(ref inner) => {
                self.popen();
                self.print_pat(inner);
                self.pclose();
            }
            PatKind::MacCall(ref m) => self.print_mac(m),
        }
        self.ann.post(self, AnnNode::Pat(pat))
    }

    fn print_arm(&mut self, arm: &ast::Arm) {
        // Note, I have no idea why this check is necessary, but here it is.
        if arm.attrs.is_empty() {
            self.s.space();
        }
        self.cbox(INDENT_UNIT);
        self.ibox(0);
        self.maybe_print_comment(arm.pat.span.lo());
        self.print_outer_attributes(&arm.attrs);
        self.print_pat(&arm.pat);
        self.s.space();
        if let Some(ref e) = arm.guard {
            self.word_space("if");
            self.print_expr(e);
            self.s.space();
        }
        self.word_space("=>");

        match arm.body.kind {
            ast::ExprKind::Block(ref blk, opt_label) => {
                if let Some(label) = opt_label {
                    self.print_ident(label.ident);
                    self.word_space(":");
                }

                // The block will close the pattern's ibox.
                self.print_block_unclosed_indent(blk);

                // If it is a user-provided unsafe block, print a comma after it.
                if let BlockCheckMode::Unsafe(ast::UserProvided) = blk.rules {
                    self.s.word(",");
                }
            }
            _ => {
                self.end(); // Close the ibox for the pattern.
                self.print_expr(&arm.body);
                self.s.word(",");
            }
        }
        self.end(); // Close enclosing cbox.
    }

    fn print_explicit_self(&mut self, explicit_self: &ast::ExplicitSelf) {
        match explicit_self.node {
            SelfKind::Value(m) => {
                self.print_mutability(m, false);
                self.s.word("self")
            }
            SelfKind::Region(ref lt, m) => {
                self.s.word("&");
                self.print_opt_lifetime(lt);
                self.print_mutability(m, false);
                self.s.word("self")
            }
            SelfKind::Explicit(ref typ, m) => {
                self.print_mutability(m, false);
                self.s.word("self");
                self.word_space(":");
                self.print_type(typ)
            }
        }
    }

    fn print_fn_full(
        &mut self,
        sig: &ast::FnSig,
        name: Ident,
        generics: &ast::Generics,
        vis: &ast::Visibility,
        defaultness: ast::Defaultness,
        body: Option<&ast::Block>,
        attrs: &[ast::Attribute],
    ) {
        if body.is_some() {
            self.head("");
        }
        self.print_visibility(vis);
        self.print_defaultness(defaultness);
        self.print_fn(&sig.decl, sig.header, Some(name), generics);
        if let Some(body) = body {
            self.nbsp();
            self.print_block_with_attrs(body, attrs);
        } else {
            self.s.word(";");
        }
    }

    crate fn print_fn(
        &mut self,
        decl: &ast::FnDecl,
        header: ast::FnHeader,
        name: Option<Ident>,
        generics: &ast::Generics,
    ) {
        self.print_fn_header_info(header);
        if let Some(name) = name {
            self.nbsp();
            self.print_ident(name);
        }
        self.print_generic_params(&generics.params);
        self.print_fn_params_and_ret(decl, false);
        self.print_where_clause(&generics.where_clause)
    }

    crate fn print_fn_params_and_ret(&mut self, decl: &ast::FnDecl, is_closure: bool) {
        let (open, close) = if is_closure { ("|", "|") } else { ("(", ")") };
        self.word(open);
        self.commasep(Inconsistent, &decl.inputs, |s, param| s.print_param(param, is_closure));
        self.word(close);
        self.print_fn_ret_ty(&decl.output)
    }

    crate fn print_movability(&mut self, movability: ast::Movability) {
        match movability {
            ast::Movability::Static => self.word_space("static"),
            ast::Movability::Movable => {}
        }
    }

    crate fn print_asyncness(&mut self, asyncness: ast::Async) {
        if asyncness.is_async() {
            self.word_nbsp("async");
        }
    }

    crate fn print_capture_clause(&mut self, capture_clause: ast::CaptureBy) {
        match capture_clause {
            ast::CaptureBy::Value => self.word_space("move"),
            ast::CaptureBy::Ref => {}
        }
    }

    pub fn print_type_bounds(&mut self, prefix: &'static str, bounds: &[ast::GenericBound]) {
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
        &mut self,
        lifetime: ast::Lifetime,
        bounds: &ast::GenericBounds,
    ) {
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
            s.print_outer_attributes_inline(&param.attrs);

            match param.kind {
                ast::GenericParamKind::Lifetime => {
                    let lt = ast::Lifetime { id: param.id, ident: param.ident };
                    s.print_lifetime_bounds(lt, &param.bounds)
                }
                ast::GenericParamKind::Type { ref default } => {
                    s.print_ident(param.ident);
                    s.print_type_bounds(":", &param.bounds);
                    if let Some(ref default) = default {
                        s.s.space();
                        s.word_space("=");
                        s.print_type(default)
                    }
                }
                ast::GenericParamKind::Const { ref ty, kw_span: _, ref default } => {
                    s.word_space("const");
                    s.print_ident(param.ident);
                    s.s.space();
                    s.word_space(":");
                    s.print_type(ty);
                    s.print_type_bounds(":", &param.bounds);
                    if let Some(ref _default) = default {
                        // FIXME(const_generics_defaults): print the `default` value here
                    }
                }
            }
        });

        self.s.word(">");
    }

    crate fn print_where_clause(&mut self, where_clause: &ast::WhereClause) {
        if where_clause.predicates.is_empty() && !where_clause.has_where_token {
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
                ast::WherePredicate::RegionPredicate(ast::WhereRegionPredicate {
                    ref lifetime,
                    ref bounds,
                    ..
                }) => {
                    self.print_lifetime_bounds(*lifetime, bounds);
                }
                ast::WherePredicate::EqPredicate(ast::WhereEqPredicate {
                    ref lhs_ty,
                    ref rhs_ty,
                    ..
                }) => {
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

    pub fn print_mutability(&mut self, mutbl: ast::Mutability, print_const: bool) {
        match mutbl {
            ast::Mutability::Mut => self.word_nbsp("mut"),
            ast::Mutability::Not => {
                if print_const {
                    self.word_nbsp("const");
                }
            }
        }
    }

    crate fn print_mt(&mut self, mt: &ast::MutTy, print_const: bool) {
        self.print_mutability(mt.mutbl, print_const);
        self.print_type(&mt.ty)
    }

    crate fn print_param(&mut self, input: &ast::Param, is_closure: bool) {
        self.ibox(INDENT_UNIT);

        self.print_outer_attributes_inline(&input.attrs);

        match input.ty.kind {
            ast::TyKind::Infer if is_closure => self.print_pat(&input.pat),
            _ => {
                if let Some(eself) = input.to_self() {
                    self.print_explicit_self(&eself);
                } else {
                    let invalid = if let PatKind::Ident(_, ident, _) = input.pat.kind {
                        ident.name == kw::Empty
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

    crate fn print_fn_ret_ty(&mut self, fn_ret_ty: &ast::FnRetTy) {
        if let ast::FnRetTy::Ty(ty) = fn_ret_ty {
            self.space_if_not_bol();
            self.ibox(INDENT_UNIT);
            self.word_space("->");
            self.print_type(ty);
            self.end();
            self.maybe_print_comment(ty.span.lo());
        }
    }

    crate fn print_ty_fn(
        &mut self,
        ext: ast::Extern,
        unsafety: ast::Unsafe,
        decl: &ast::FnDecl,
        name: Option<Ident>,
        generic_params: &[ast::GenericParam],
    ) {
        self.ibox(INDENT_UNIT);
        if !generic_params.is_empty() {
            self.s.word("for");
            self.print_generic_params(generic_params);
        }
        let generics = ast::Generics {
            params: Vec::new(),
            where_clause: ast::WhereClause {
                has_where_token: false,
                predicates: Vec::new(),
                span: rustc_span::DUMMY_SP,
            },
            span: rustc_span::DUMMY_SP,
        };
        let header = ast::FnHeader { unsafety, ext, ..ast::FnHeader::default() };
        self.print_fn(decl, header, name, &generics);
        self.end();
    }

    crate fn maybe_print_trailing_comment(
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

    crate fn print_fn_header_info(&mut self, header: ast::FnHeader) {
        self.print_constness(header.constness);
        self.print_asyncness(header.asyncness);
        self.print_unsafety(header.unsafety);

        match header.ext {
            ast::Extern::None => {}
            ast::Extern::Implicit => {
                self.word_nbsp("extern");
            }
            ast::Extern::Explicit(abi) => {
                self.word_nbsp("extern");
                self.print_literal(&abi.as_lit());
                self.nbsp();
            }
        }

        self.s.word("fn")
    }

    crate fn print_unsafety(&mut self, s: ast::Unsafe) {
        match s {
            ast::Unsafe::No => {}
            ast::Unsafe::Yes(_) => self.word_nbsp("unsafe"),
        }
    }

    crate fn print_constness(&mut self, s: ast::Const) {
        match s {
            ast::Const::No => {}
            ast::Const::Yes(_) => self.word_nbsp("const"),
        }
    }

    crate fn print_is_auto(&mut self, s: ast::IsAuto) {
        match s {
            ast::IsAuto::Yes => self.word_nbsp("auto"),
            ast::IsAuto::No => {}
        }
    }
}
