mod delimited;
mod expr;
mod item;

use crate::pp::Breaks::{Consistent, Inconsistent};
use crate::pp::{self, Breaks};

use rustc_ast::ptr::P;
use rustc_ast::token::{self, BinOpToken, CommentKind, Delimiter, Nonterminal, Token, TokenKind};
use rustc_ast::tokenstream::{TokenStream, TokenTree};
use rustc_ast::util::classify;
use rustc_ast::util::comments::{gather_comments, Comment, CommentStyle};
use rustc_ast::util::parser;
use rustc_ast::{self as ast, AttrArgs, AttrArgsEq, BlockCheckMode, PatKind};
use rustc_ast::{attr, BindingAnnotation, ByRef, DelimArgs, RangeEnd, RangeSyntax, Term};
use rustc_ast::{GenericArg, GenericBound, SelfKind, TraitBoundModifier};
use rustc_ast::{InlineAsmOperand, InlineAsmRegOrRegClass};
use rustc_ast::{InlineAsmOptions, InlineAsmTemplatePiece};
use rustc_span::edition::Edition;
use rustc_span::source_map::{SourceMap, Spanned};
use rustc_span::symbol::{kw, sym, Ident, IdentPrinter, Symbol};
use rustc_span::{BytePos, FileName, Span, DUMMY_SP};

use rustc_ast::attr::AttrIdGenerator;
use std::borrow::Cow;

pub use self::delimited::IterDelimited;

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

    // FIXME: This shouldn't probably clone lmao
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
}

pub(crate) const INDENT_UNIT: isize = 4;

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
    g: &AttrIdGenerator,
) -> String {
    let mut s =
        State { s: pp::Printer::new(), comments: Some(Comments::new(sm, filename, input)), ann };

    if is_expanded && !krate.attrs.iter().any(|attr| attr.has_name(sym::no_core)) {
        // We need to print `#![no_std]` (and its feature gate) so that
        // compiling pretty-printed source won't inject libstd again.
        // However, we don't want these attributes in the AST because
        // of the feature gate, so we fake them up here.

        // `#![feature(prelude_import)]`
        let fake_attr = attr::mk_attr_nested_word(
            g,
            ast::AttrStyle::Inner,
            sym::feature,
            sym::prelude_import,
            DUMMY_SP,
        );
        s.print_attribute(&fake_attr);

        // Currently, in Rust 2018 we don't have `extern crate std;` at the crate
        // root, so this is not needed, and actually breaks things.
        if edition.is_rust_2015() {
            // `#![no_std]`
            let fake_attr = attr::mk_attr_word(g, ast::AttrStyle::Inner, sym::no_std, DUMMY_SP);
            s.print_attribute(&fake_attr);
        }
    }

    s.print_inner_attributes(&krate.attrs);
    for item in &krate.items {
        s.print_item(item);
    }
    s.print_remaining_comments();
    s.ann.post(&mut s, AnnNode::Crate(krate));
    s.s.eof()
}

/// This makes printed token streams look slightly nicer,
/// and also addresses some specific regressions described in #63896 and #73345.
fn tt_prepend_space(tt: &TokenTree, prev: &TokenTree) -> bool {
    if let TokenTree::Token(token, _) = prev {
        if matches!(token.kind, token::Dot | token::Dollar) {
            return false;
        }
        if let token::DocComment(comment_kind, ..) = token.kind {
            return comment_kind != CommentKind::Line;
        }
    }
    match tt {
        TokenTree::Token(token, _) => !matches!(token.kind, token::Comma | token::Not | token::Dot),
        TokenTree::Delimited(_, Delimiter::Parenthesis, _) => {
            !matches!(prev, TokenTree::Token(Token { kind: token::Ident(..), .. }, _))
        }
        TokenTree::Delimited(_, Delimiter::Bracket, _) => {
            !matches!(prev, TokenTree::Token(Token { kind: token::Pound, .. }, _))
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
        (CommentKind::Line, ast::AttrStyle::Outer) => format!("///{data}"),
        (CommentKind::Line, ast::AttrStyle::Inner) => format!("//!{data}"),
        (CommentKind::Block, ast::AttrStyle::Outer) => format!("/**{data}*/"),
        (CommentKind::Block, ast::AttrStyle::Inner) => format!("/*!{data}*/"),
    }
}

pub fn literal_to_string(lit: token::Lit) -> String {
    let token::Lit { kind, symbol, suffix } = lit;
    let mut out = match kind {
        token::Byte => format!("b'{symbol}'"),
        token::Char => format!("'{symbol}'"),
        token::Str => format!("\"{symbol}\""),
        token::StrRaw(n) => {
            format!("r{delim}\"{string}\"{delim}", delim = "#".repeat(n as usize), string = symbol)
        }
        token::ByteStr => format!("b\"{symbol}\""),
        token::ByteStrRaw(n) => {
            format!("br{delim}\"{string}\"{delim}", delim = "#".repeat(n as usize), string = symbol)
        }
        token::Integer | token::Float | token::Bool | token::Err => symbol.to_string(),
    };

    if let Some(suffix) = suffix {
        out.push_str(suffix.as_str())
    }

    out
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

    fn maybe_print_comment(&mut self, pos: BytePos) -> bool {
        let mut has_comment = false;
        while let Some(cmnt) = self.next_comment() {
            if cmnt.pos < pos {
                has_comment = true;
                self.print_comment(&cmnt);
            } else {
                break;
            }
        }
        has_comment
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
                    self.visual_align();
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
                    Some(pp::Token::String(s)) => ";" == s,
                    Some(pp::Token::Begin(_)) => true,
                    Some(pp::Token::End) => true,
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

    fn maybe_print_trailing_comment(&mut self, span: rustc_span::Span, next_pos: Option<BytePos>) {
        if let Some(cmnts) = self.comments() {
            if let Some(cmnt) = cmnts.trailing_comment(span, next_pos) {
                self.print_comment(&cmnt);
            }
        }
    }

    fn print_remaining_comments(&mut self) {
        // If there aren't any remaining comments, then we need to manually
        // make sure there is a line break at the end.
        if self.next_comment().is_none() {
            self.hardbreak();
        }
        while let Some(cmnt) = self.next_comment() {
            self.print_comment(&cmnt)
        }
    }

    fn print_meta_item_lit(&mut self, lit: &ast::MetaItemLit) {
        self.print_token_literal(lit.as_token_lit(), lit.span)
    }

    fn print_token_literal(&mut self, token_lit: token::Lit, span: Span) {
        self.maybe_print_comment(span.lo());
        self.word(token_lit.to_string())
    }

    fn print_string(&mut self, st: &str, style: ast::StrStyle) {
        let st = match style {
            ast::StrStyle::Cooked => format!("\"{}\"", st.escape_debug()),
            ast::StrStyle::Raw(n) => {
                format!("r{delim}\"{string}\"{delim}", delim = "#".repeat(n as usize), string = st)
            }
        };
        self.word(st)
    }

    fn print_symbol(&mut self, sym: Symbol, style: ast::StrStyle) {
        self.print_string(sym.as_str(), style);
    }

    fn print_inner_attributes(&mut self, attrs: &[ast::Attribute]) -> bool {
        self.print_either_attributes(attrs, ast::AttrStyle::Inner, false, true)
    }

    fn print_inner_attributes_no_trailing_hardbreak(&mut self, attrs: &[ast::Attribute]) -> bool {
        self.print_either_attributes(attrs, ast::AttrStyle::Inner, false, false)
    }

    fn print_outer_attributes(&mut self, attrs: &[ast::Attribute]) -> bool {
        self.print_either_attributes(attrs, ast::AttrStyle::Outer, false, true)
    }

    fn print_inner_attributes_inline(&mut self, attrs: &[ast::Attribute]) -> bool {
        self.print_either_attributes(attrs, ast::AttrStyle::Inner, true, true)
    }

    fn print_outer_attributes_inline(&mut self, attrs: &[ast::Attribute]) -> bool {
        self.print_either_attributes(attrs, ast::AttrStyle::Outer, true, true)
    }

    fn print_either_attributes(
        &mut self,
        attrs: &[ast::Attribute],
        kind: ast::AttrStyle,
        is_inline: bool,
        trailing_hardbreak: bool,
    ) -> bool {
        let mut printed = false;
        for attr in attrs {
            if attr.style == kind {
                self.print_attribute_inline(attr, is_inline);
                if is_inline {
                    self.nbsp();
                }
                printed = true;
            }
        }
        if printed && trailing_hardbreak && !is_inline {
            self.hardbreak_if_not_bol();
        }
        printed
    }

    fn print_attribute(&mut self, attr: &ast::Attribute) {
        self.print_attribute_inline(attr, false)
    }

    fn print_attribute_inline(&mut self, attr: &ast::Attribute, is_inline: bool) {
        if !is_inline {
            self.hardbreak_if_not_bol();
        }
        self.maybe_print_comment(attr.span.lo());
        match &attr.kind {
            ast::AttrKind::Normal(normal) => {
                match attr.style {
                    ast::AttrStyle::Inner => self.word("#!["),
                    ast::AttrStyle::Outer => self.word("#["),
                }
                self.print_attr_item(&normal.item, attr.span);
                self.word("]");
            }
            ast::AttrKind::DocComment(comment_kind, data) => {
                self.word(doc_comment_to_string(*comment_kind, attr.style, *data));
                self.hardbreak()
            }
        }
    }

    fn print_attr_item(&mut self, item: &ast::AttrItem, span: Span) {
        self.ibox(0);
        match &item.args {
            AttrArgs::Delimited(DelimArgs { dspan: _, delim, tokens }) => self.print_mac_common(
                Some(MacHeader::Path(&item.path)),
                false,
                None,
                delim.to_token(),
                tokens,
                true,
                span,
            ),
            AttrArgs::Empty => {
                self.print_path(&item.path, false, 0);
            }
            AttrArgs::Eq(_, AttrArgsEq::Ast(expr)) => {
                self.print_path(&item.path, false, 0);
                self.space();
                self.word_space("=");
                let token_str = self.expr_to_string(expr);
                self.word(token_str);
            }
            AttrArgs::Eq(_, AttrArgsEq::Hir(lit)) => {
                self.print_path(&item.path, false, 0);
                self.space();
                self.word_space("=");
                let token_str = self.meta_item_lit_to_string(lit);
                self.word(token_str);
            }
        }
        self.end();
    }

    fn print_meta_list_item(&mut self, item: &ast::NestedMetaItem) {
        match item {
            ast::NestedMetaItem::MetaItem(mi) => self.print_meta_item(mi),
            ast::NestedMetaItem::Lit(lit) => self.print_meta_item_lit(lit),
        }
    }

    fn print_meta_item(&mut self, item: &ast::MetaItem) {
        self.ibox(INDENT_UNIT);
        match &item.kind {
            ast::MetaItemKind::Word => self.print_path(&item.path, false, 0),
            ast::MetaItemKind::NameValue(value) => {
                self.print_path(&item.path, false, 0);
                self.space();
                self.word_space("=");
                self.print_meta_item_lit(value);
            }
            ast::MetaItemKind::List(items) => {
                self.print_path(&item.path, false, 0);
                self.popen();
                self.commasep(Consistent, items, |s, i| s.print_meta_list_item(i));
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
            TokenTree::Token(token, _) => {
                let token_str = self.token_to_string_ext(token, convert_dollar_crate);
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
            self.print_tt(tt, convert_dollar_crate);
            if let Some(next) = iter.peek() {
                if tt_prepend_space(next, tt) {
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
        delim: Delimiter,
        tts: &TokenStream,
        convert_dollar_crate: bool,
        span: Span,
    ) {
        if delim == Delimiter::Brace {
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
            Delimiter::Brace => {
                if header.is_some() || has_bang || ident.is_some() {
                    self.nbsp();
                }
                self.word("{");
                if !tts.is_empty() {
                    self.space();
                }
                self.ibox(0);
                self.print_tts(tts, convert_dollar_crate);
                self.end();
                let empty = tts.is_empty();
                self.bclose(span, empty);
            }
            delim => {
                let token_str = self.token_kind_to_string(&token::OpenDelim(delim));
                self.word(token_str);
                self.ibox(0);
                self.print_tts(tts, convert_dollar_crate);
                self.end();
                let token_str = self.token_kind_to_string(&token::CloseDelim(delim));
                self.word(token_str);
            }
        }
    }

    fn print_mac_def(
        &mut self,
        macro_def: &ast::MacroDef,
        ident: &Ident,
        sp: Span,
        print_visibility: impl FnOnce(&mut Self),
    ) {
        let (kw, has_bang) = if macro_def.macro_rules {
            ("macro_rules", true)
        } else {
            print_visibility(self);
            ("macro", false)
        };
        self.print_mac_common(
            Some(MacHeader::Keyword(kw)),
            has_bang,
            Some(*ident),
            macro_def.body.delim.to_token(),
            &macro_def.body.tokens.clone(),
            true,
            sp,
        );
        if macro_def.body.need_semicolon() {
            self.word(";");
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
            if let Some(args) = &segment.args {
                self.print_generic_args(args, colons_before_params);
            }
        }
    }

    fn head<S: Into<Cow<'static, str>>>(&mut self, w: S) {
        let w = w.into();
        // Outer-box is consistent.
        self.cbox(INDENT_UNIT);
        // Head-box is inconsistent.
        self.ibox(0);
        // Keyword that starts the head.
        if !w.is_empty() {
            self.word_nbsp(w);
        }
    }

    fn bopen(&mut self) {
        self.word("{");
        self.end(); // Close the head-box.
    }

    fn bclose_maybe_open(&mut self, span: rustc_span::Span, empty: bool, close_box: bool) {
        let has_comment = self.maybe_print_comment(span.hi());
        if !empty || has_comment {
            self.break_offset_if_not_bol(1, -(INDENT_UNIT as isize));
        }
        self.word("}");
        if close_box {
            self.end(); // Close the outer-box.
        }
    }

    fn bclose(&mut self, span: rustc_span::Span, empty: bool) {
        let close_box = true;
        self.bclose_maybe_open(span, empty, close_box)
    }

    fn break_offset_if_not_bol(&mut self, n: usize, off: isize) {
        if !self.is_beginning_of_line() {
            self.break_offset(n, off)
        } else if off != 0 {
            if let Some(last_token) = self.last_token_still_buffered() {
                if last_token.is_hardbreak_tok() {
                    // We do something pretty sketchy here: tuck the nonzero
                    // offset-adjustment we were going to deposit along with the
                    // break into the previous hardbreak.
                    self.replace_last_token_still_buffered(pp::Printer::hardbreak_tok_offset(off));
                }
            }
        }
    }

    fn nonterminal_to_string(&self, nt: &Nonterminal) -> String {
        match nt {
            token::NtExpr(e) => self.expr_to_string(e),
            token::NtMeta(e) => self.attr_item_to_string(e),
            token::NtTy(e) => self.ty_to_string(e),
            token::NtPath(e) => self.path_to_string(e),
            token::NtItem(e) => self.item_to_string(e),
            token::NtBlock(e) => self.block_to_string(e),
            token::NtStmt(e) => self.stmt_to_string(e),
            token::NtPat(e) => self.pat_to_string(e),
            token::NtIdent(e, is_raw) => IdentPrinter::for_ast_ident(*e, *is_raw).to_string(),
            token::NtLifetime(e) => e.to_string(),
            token::NtLiteral(e) => self.expr_to_string(e),
            token::NtVis(e) => self.vis_to_string(e),
        }
    }

    /// Print the token kind precisely, without converting `$crate` into its respective crate name.
    fn token_kind_to_string(&self, tok: &TokenKind) -> Cow<'static, str> {
        self.token_kind_to_string_ext(tok, None)
    }

    fn token_kind_to_string_ext(
        &self,
        tok: &TokenKind,
        convert_dollar_crate: Option<Span>,
    ) -> Cow<'static, str> {
        match *tok {
            token::Eq => "=".into(),
            token::Lt => "<".into(),
            token::Le => "<=".into(),
            token::EqEq => "==".into(),
            token::Ne => "!=".into(),
            token::Ge => ">=".into(),
            token::Gt => ">".into(),
            token::Not => "!".into(),
            token::Tilde => "~".into(),
            token::OrOr => "||".into(),
            token::AndAnd => "&&".into(),
            token::BinOp(op) => binop_to_string(op).into(),
            token::BinOpEq(op) => format!("{}=", binop_to_string(op)).into(),

            /* Structural symbols */
            token::At => "@".into(),
            token::Dot => ".".into(),
            token::DotDot => "..".into(),
            token::DotDotDot => "...".into(),
            token::DotDotEq => "..=".into(),
            token::Comma => ",".into(),
            token::Semi => ";".into(),
            token::Colon => ":".into(),
            token::ModSep => "::".into(),
            token::RArrow => "->".into(),
            token::LArrow => "<-".into(),
            token::FatArrow => "=>".into(),
            token::OpenDelim(Delimiter::Parenthesis) => "(".into(),
            token::CloseDelim(Delimiter::Parenthesis) => ")".into(),
            token::OpenDelim(Delimiter::Bracket) => "[".into(),
            token::CloseDelim(Delimiter::Bracket) => "]".into(),
            token::OpenDelim(Delimiter::Brace) => "{".into(),
            token::CloseDelim(Delimiter::Brace) => "}".into(),
            token::OpenDelim(Delimiter::Invisible) | token::CloseDelim(Delimiter::Invisible) => {
                "".into()
            }
            token::Pound => "#".into(),
            token::Dollar => "$".into(),
            token::Question => "?".into(),
            token::SingleQuote => "'".into(),

            /* Literals */
            token::Literal(lit) => literal_to_string(lit).into(),

            /* Name components */
            token::Ident(s, is_raw) => {
                IdentPrinter::new(s, is_raw, convert_dollar_crate).to_string().into()
            }
            token::Lifetime(s) => s.to_string().into(),

            /* Other */
            token::DocComment(comment_kind, attr_style, data) => {
                doc_comment_to_string(comment_kind, attr_style, data).into()
            }
            token::Eof => "<eof>".into(),

            token::Interpolated(ref nt) => self.nonterminal_to_string(nt).into(),
        }
    }

    /// Print the token precisely, without converting `$crate` into its respective crate name.
    fn token_to_string(&self, token: &Token) -> Cow<'static, str> {
        self.token_to_string_ext(token, false)
    }

    fn token_to_string_ext(&self, token: &Token, convert_dollar_crate: bool) -> Cow<'static, str> {
        let convert_dollar_crate = convert_dollar_crate.then_some(token.span);
        self.token_kind_to_string_ext(&token.kind, convert_dollar_crate)
    }

    fn ty_to_string(&self, ty: &ast::Ty) -> String {
        Self::to_string(|s| s.print_type(ty))
    }

    fn bounds_to_string(&self, bounds: &[ast::GenericBound]) -> String {
        Self::to_string(|s| s.print_type_bounds(bounds))
    }

    fn pat_to_string(&self, pat: &ast::Pat) -> String {
        Self::to_string(|s| s.print_pat(pat))
    }

    fn expr_to_string(&self, e: &ast::Expr) -> String {
        Self::to_string(|s| s.print_expr(e))
    }

    fn meta_item_lit_to_string(&self, lit: &ast::MetaItemLit) -> String {
        Self::to_string(|s| s.print_meta_item_lit(lit))
    }

    fn tt_to_string(&self, tt: &TokenTree) -> String {
        Self::to_string(|s| s.print_tt(tt, false))
    }

    fn tts_to_string(&self, tokens: &TokenStream) -> String {
        Self::to_string(|s| s.print_tts(tokens, false))
    }

    fn stmt_to_string(&self, stmt: &ast::Stmt) -> String {
        Self::to_string(|s| s.print_stmt(stmt))
    }

    fn item_to_string(&self, i: &ast::Item) -> String {
        Self::to_string(|s| s.print_item(i))
    }

    fn assoc_item_to_string(&self, i: &ast::AssocItem) -> String {
        Self::to_string(|s| s.print_assoc_item(i))
    }

    fn foreign_item_to_string(&self, i: &ast::ForeignItem) -> String {
        Self::to_string(|s| s.print_foreign_item(i))
    }

    fn generic_params_to_string(&self, generic_params: &[ast::GenericParam]) -> String {
        Self::to_string(|s| s.print_generic_params(generic_params))
    }

    fn path_to_string(&self, p: &ast::Path) -> String {
        Self::to_string(|s| s.print_path(p, false, 0))
    }

    fn path_segment_to_string(&self, p: &ast::PathSegment) -> String {
        Self::to_string(|s| s.print_path_segment(p, false))
    }

    fn vis_to_string(&self, v: &ast::Visibility) -> String {
        Self::to_string(|s| s.print_visibility(v))
    }

    fn block_to_string(&self, blk: &ast::Block) -> String {
        Self::to_string(|s| {
            // Containing cbox, will be closed by `print_block` at `}`.
            s.cbox(INDENT_UNIT);
            // Head-ibox, will be closed by `print_block` after `{`.
            s.ibox(0);
            s.print_block(blk)
        })
    }

    fn meta_list_item_to_string(&self, li: &ast::NestedMetaItem) -> String {
        Self::to_string(|s| s.print_meta_list_item(li))
    }

    fn attr_item_to_string(&self, ai: &ast::AttrItem) -> String {
        Self::to_string(|s| s.print_attr_item(ai, ai.path.span))
    }

    fn attribute_to_string(&self, attr: &ast::Attribute) -> String {
        Self::to_string(|s| s.print_attribute(attr))
    }

    fn param_to_string(&self, arg: &ast::Param) -> String {
        Self::to_string(|s| s.print_param(arg, false))
    }

    fn to_string(f: impl FnOnce(&mut State<'_>)) -> String {
        let mut printer = State::new();
        f(&mut printer);
        printer.s.eof()
    }
}

impl<'a> PrintState<'a> for State<'a> {
    fn comments(&mut self) -> &mut Option<Comments<'a>> {
        &mut self.comments
    }

    fn print_ident(&mut self, ident: Ident) {
        self.word(IdentPrinter::for_ast_ident(ident, ident.is_raw_guess()).to_string());
        self.ann.post(self, AnnNode::Ident(&ident))
    }

    fn print_generic_args(&mut self, args: &ast::GenericArgs, colons_before_params: bool) {
        if colons_before_params {
            self.word("::")
        }

        match args {
            ast::GenericArgs::AngleBracketed(data) => {
                self.word("<");
                self.commasep(Inconsistent, &data.args, |s, arg| match arg {
                    ast::AngleBracketedArg::Arg(a) => s.print_generic_arg(a),
                    ast::AngleBracketedArg::Constraint(c) => s.print_assoc_constraint(c),
                });
                self.word(">")
            }

            ast::GenericArgs::Parenthesized(data) => {
                self.word("(");
                self.commasep(Inconsistent, &data.inputs, |s, ty| s.print_type(ty));
                self.word(")");
                self.print_fn_ret_ty(&data.output);
            }
        }
    }
}

impl<'a> State<'a> {
    pub fn new() -> State<'a> {
        State { s: pp::Printer::new(), comments: None, ann: &NoAnn }
    }

    pub(crate) fn commasep_cmnt<T, F, G>(
        &mut self,
        b: Breaks,
        elts: &[T],
        mut op: F,
        mut get_span: G,
    ) where
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
                self.word(",");
                self.maybe_print_trailing_comment(get_span(elt), Some(get_span(&elts[i]).hi()));
                self.space_if_not_bol();
            }
        }
        self.end();
    }

    pub(crate) fn commasep_exprs(&mut self, b: Breaks, exprs: &[P<ast::Expr>]) {
        self.commasep_cmnt(b, exprs, |s, e| s.print_expr(e), |e| e.span)
    }

    pub fn print_opt_lifetime(&mut self, lifetime: &Option<ast::Lifetime>) {
        if let Some(lt) = *lifetime {
            self.print_lifetime(lt);
            self.nbsp();
        }
    }

    pub fn print_assoc_constraint(&mut self, constraint: &ast::AssocConstraint) {
        self.print_ident(constraint.ident);
        constraint.gen_args.as_ref().map(|args| self.print_generic_args(args, false));
        self.space();
        match &constraint.kind {
            ast::AssocConstraintKind::Equality { term } => {
                self.word_space("=");
                match term {
                    Term::Ty(ty) => self.print_type(ty),
                    Term::Const(c) => self.print_expr_anon_const(c, &[]),
                }
            }
            ast::AssocConstraintKind::Bound { bounds } => {
                if !bounds.is_empty() {
                    self.word_nbsp(":");
                    self.print_type_bounds(bounds);
                }
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
        match &ty.kind {
            ast::TyKind::Slice(ty) => {
                self.word("[");
                self.print_type(ty);
                self.word("]");
            }
            ast::TyKind::Ptr(mt) => {
                self.word("*");
                self.print_mt(mt, true);
            }
            ast::TyKind::Ref(lifetime, mt) => {
                self.word("&");
                self.print_opt_lifetime(lifetime);
                self.print_mt(mt, false);
            }
            ast::TyKind::Never => {
                self.word("!");
            }
            ast::TyKind::Tup(elts) => {
                self.popen();
                self.commasep(Inconsistent, elts, |s, ty| s.print_type(ty));
                if elts.len() == 1 {
                    self.word(",");
                }
                self.pclose();
            }
            ast::TyKind::Paren(typ) => {
                self.popen();
                self.print_type(typ);
                self.pclose();
            }
            ast::TyKind::BareFn(f) => {
                self.print_ty_fn(f.ext, f.unsafety, &f.decl, None, &f.generic_params);
            }
            ast::TyKind::Path(None, path) => {
                self.print_path(path, false, 0);
            }
            ast::TyKind::Path(Some(qself), path) => self.print_qpath(path, qself, false),
            ast::TyKind::TraitObject(bounds, syntax) => {
                if *syntax == ast::TraitObjectSyntax::Dyn {
                    self.word_nbsp("dyn");
                }
                self.print_type_bounds(bounds);
            }
            ast::TyKind::ImplTrait(_, bounds) => {
                self.word_nbsp("impl");
                self.print_type_bounds(bounds);
            }
            ast::TyKind::Array(ty, length) => {
                self.word("[");
                self.print_type(ty);
                self.word("; ");
                self.print_expr(&length.value);
                self.word("]");
            }
            ast::TyKind::Typeof(e) => {
                self.word("typeof(");
                self.print_expr(&e.value);
                self.word(")");
            }
            ast::TyKind::Infer => {
                self.word("_");
            }
            ast::TyKind::Err => {
                self.popen();
                self.word("/*ERROR*/");
                self.pclose();
            }
            ast::TyKind::ImplicitSelf => {
                self.word("Self");
            }
            ast::TyKind::MacCall(m) => {
                self.print_mac(m);
            }
            ast::TyKind::CVarArgs => {
                self.word("...");
            }
        }
        self.end();
    }

    fn print_trait_ref(&mut self, t: &ast::TraitRef) {
        self.print_path(&t.path, false, 0)
    }

    fn print_formal_generic_params(&mut self, generic_params: &[ast::GenericParam]) {
        if !generic_params.is_empty() {
            self.word("for");
            self.print_generic_params(generic_params);
            self.nbsp();
        }
    }

    fn print_poly_trait_ref(&mut self, t: &ast::PolyTraitRef) {
        self.print_formal_generic_params(&t.bound_generic_params);
        self.print_trait_ref(&t.trait_ref)
    }

    pub(crate) fn print_stmt(&mut self, st: &ast::Stmt) {
        self.maybe_print_comment(st.span.lo());
        match &st.kind {
            ast::StmtKind::Local(loc) => {
                self.print_outer_attributes(&loc.attrs);
                self.space_if_not_bol();
                self.ibox(INDENT_UNIT);
                self.word_nbsp("let");

                self.ibox(INDENT_UNIT);
                self.print_local_decl(loc);
                self.end();
                if let Some((init, els)) = loc.kind.init_else_opt() {
                    self.nbsp();
                    self.word_space("=");
                    self.print_expr(init);
                    if let Some(els) = els {
                        self.cbox(INDENT_UNIT);
                        self.ibox(INDENT_UNIT);
                        self.word(" else ");
                        self.print_block(els);
                    }
                }
                self.word(";");
                self.end(); // `let` ibox
            }
            ast::StmtKind::Item(item) => self.print_item(item),
            ast::StmtKind::Expr(expr) => {
                self.space_if_not_bol();
                self.print_expr_outer_attr_style(expr, false);
                if classify::expr_requires_semi_to_be_stmt(expr) {
                    self.word(";");
                }
            }
            ast::StmtKind::Semi(expr) => {
                self.space_if_not_bol();
                self.print_expr_outer_attr_style(expr, false);
                self.word(";");
            }
            ast::StmtKind::Empty => {
                self.space_if_not_bol();
                self.word(";");
            }
            ast::StmtKind::MacCall(mac) => {
                self.space_if_not_bol();
                self.print_outer_attributes(&mac.attrs);
                self.print_mac(&mac.mac);
                if mac.style == ast::MacStmtStyle::Semicolon {
                    self.word(";");
                }
            }
        }
        self.maybe_print_trailing_comment(st.span, None)
    }

    pub(crate) fn print_block(&mut self, blk: &ast::Block) {
        self.print_block_with_attrs(blk, &[])
    }

    pub(crate) fn print_block_unclosed_indent(&mut self, blk: &ast::Block) {
        self.print_block_maybe_unclosed(blk, &[], false)
    }

    pub(crate) fn print_block_with_attrs(&mut self, blk: &ast::Block, attrs: &[ast::Attribute]) {
        self.print_block_maybe_unclosed(blk, attrs, true)
    }

    pub(crate) fn print_block_maybe_unclosed(
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

        let has_attrs = self.print_inner_attributes(attrs);

        for (i, st) in blk.stmts.iter().enumerate() {
            match &st.kind {
                ast::StmtKind::Expr(expr) if i == blk.stmts.len() - 1 => {
                    self.maybe_print_comment(st.span.lo());
                    self.space_if_not_bol();
                    self.print_expr_outer_attr_style(expr, false);
                    self.maybe_print_trailing_comment(expr.span, Some(blk.span.hi()));
                }
                _ => self.print_stmt(st),
            }
        }

        let empty = !has_attrs && blk.stmts.is_empty();
        self.bclose_maybe_open(blk.span, empty, close_box);
        self.ann.post(self, AnnNode::Block(blk))
    }

    /// Print a `let pat = expr` expression.
    pub(crate) fn print_let(&mut self, pat: &ast::Pat, expr: &ast::Expr) {
        self.word("let ");
        self.print_pat(pat);
        self.space();
        self.word_space("=");
        let npals = || parser::needs_par_as_let_scrutinee(expr.precedence().order());
        self.print_expr_cond_paren(expr, Self::cond_needs_par(expr) || npals())
    }

    pub(crate) fn print_mac(&mut self, m: &ast::MacCall) {
        self.print_mac_common(
            Some(MacHeader::Path(&m.path)),
            true,
            None,
            m.args.delim.to_token(),
            &m.args.tokens.clone(),
            true,
            m.span(),
        );
    }

    fn print_inline_asm(&mut self, asm: &ast::InlineAsm) {
        enum AsmArg<'a> {
            Template(String),
            Operand(&'a InlineAsmOperand),
            ClobberAbi(Symbol),
            Options(InlineAsmOptions),
        }

        let mut args = vec![AsmArg::Template(InlineAsmTemplatePiece::to_string(&asm.template))];
        args.extend(asm.operands.iter().map(|(o, _)| AsmArg::Operand(o)));
        for (abi, _) in &asm.clobber_abis {
            args.push(AsmArg::ClobberAbi(*abi));
        }
        if !asm.options.is_empty() {
            args.push(AsmArg::Options(asm.options));
        }

        self.popen();
        self.commasep(Consistent, &args, |s, arg| match arg {
            AsmArg::Template(template) => s.print_string(template, ast::StrStyle::Cooked),
            AsmArg::Operand(op) => {
                let print_reg_or_class = |s: &mut Self, r: &InlineAsmRegOrRegClass| match r {
                    InlineAsmRegOrRegClass::Reg(r) => s.print_symbol(*r, ast::StrStyle::Cooked),
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
                    InlineAsmOperand::Const { anon_const } => {
                        s.word("const");
                        s.space();
                        s.print_expr(&anon_const.value);
                    }
                    InlineAsmOperand::Sym { sym } => {
                        s.word("sym");
                        s.space();
                        if let Some(qself) = &sym.qself {
                            s.print_qpath(&sym.path, qself, true);
                        } else {
                            s.print_path(&sym.path, true, 0);
                        }
                    }
                }
            }
            AsmArg::ClobberAbi(abi) => {
                s.word("clobber_abi");
                s.popen();
                s.print_symbol(*abi, ast::StrStyle::Cooked);
                s.pclose();
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
                if opts.contains(InlineAsmOptions::RAW) {
                    options.push("raw");
                }
                if opts.contains(InlineAsmOptions::MAY_UNWIND) {
                    options.push("may_unwind");
                }
                s.commasep(Inconsistent, &options, |s, &opt| {
                    s.word(opt);
                });
                s.pclose();
            }
        });
        self.pclose();
    }

    pub(crate) fn print_local_decl(&mut self, loc: &ast::Local) {
        self.print_pat(&loc.pat);
        if let Some(ty) = &loc.ty {
            self.word_space(":");
            self.print_type(ty);
        }
    }

    pub(crate) fn print_name(&mut self, name: Symbol) {
        self.word(name.to_string());
        self.ann.post(self, AnnNode::Name(&name))
    }

    fn print_qpath(&mut self, path: &ast::Path, qself: &ast::QSelf, colons_before_params: bool) {
        self.word("<");
        self.print_type(&qself.ty);
        if qself.position > 0 {
            self.space();
            self.word_space("as");
            let depth = path.segments.len() - qself.position;
            self.print_path(path, false, depth);
        }
        self.word(">");
        for item_segment in &path.segments[qself.position..] {
            self.word("::");
            self.print_ident(item_segment.ident);
            if let Some(args) = &item_segment.args {
                self.print_generic_args(args, colons_before_params)
            }
        }
    }

    pub(crate) fn print_pat(&mut self, pat: &ast::Pat) {
        self.maybe_print_comment(pat.span.lo());
        self.ann.pre(self, AnnNode::Pat(pat));
        /* Pat isn't normalized, but the beauty of it
        is that it doesn't matter */
        match &pat.kind {
            PatKind::Wild => self.word("_"),
            PatKind::Ident(BindingAnnotation(by_ref, mutbl), ident, sub) => {
                if *by_ref == ByRef::Yes {
                    self.word_nbsp("ref");
                }
                if mutbl.is_mut() {
                    self.word_nbsp("mut");
                }
                self.print_ident(*ident);
                if let Some(p) = sub {
                    self.space();
                    self.word_space("@");
                    self.print_pat(p);
                }
            }
            PatKind::TupleStruct(qself, path, elts) => {
                if let Some(qself) = qself {
                    self.print_qpath(path, qself, true);
                } else {
                    self.print_path(path, true, 0);
                }
                self.popen();
                self.commasep(Inconsistent, elts, |s, p| s.print_pat(p));
                self.pclose();
            }
            PatKind::Or(pats) => {
                self.strsep("|", true, Inconsistent, pats, |s, p| s.print_pat(p));
            }
            PatKind::Path(None, path) => {
                self.print_path(path, true, 0);
            }
            PatKind::Path(Some(qself), path) => {
                self.print_qpath(path, qself, false);
            }
            PatKind::Struct(qself, path, fields, etc) => {
                if let Some(qself) = qself {
                    self.print_qpath(path, qself, true);
                } else {
                    self.print_path(path, true, 0);
                }
                self.nbsp();
                self.word("{");
                let empty = fields.is_empty() && !etc;
                if !empty {
                    self.space();
                }
                self.commasep_cmnt(
                    Consistent,
                    fields,
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
                if *etc {
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
            PatKind::Tuple(elts) => {
                self.popen();
                self.commasep(Inconsistent, elts, |s, p| s.print_pat(p));
                if elts.len() == 1 {
                    self.word(",");
                }
                self.pclose();
            }
            PatKind::Box(inner) => {
                self.word("box ");
                self.print_pat(inner);
            }
            PatKind::Ref(inner, mutbl) => {
                self.word("&");
                if mutbl.is_mut() {
                    self.word("mut ");
                }
                if let PatKind::Ident(ast::BindingAnnotation::MUT, ..) = inner.kind {
                    self.popen();
                    self.print_pat(inner);
                    self.pclose();
                } else {
                    self.print_pat(inner);
                }
            }
            PatKind::Lit(e) => self.print_expr(e),
            PatKind::Range(begin, end, Spanned { node: end_kind, .. }) => {
                if let Some(e) = begin {
                    self.print_expr(e);
                }
                match end_kind {
                    RangeEnd::Included(RangeSyntax::DotDotDot) => self.word("..."),
                    RangeEnd::Included(RangeSyntax::DotDotEq) => self.word("..="),
                    RangeEnd::Excluded => self.word(".."),
                }
                if let Some(e) = end {
                    self.print_expr(e);
                }
            }
            PatKind::Slice(elts) => {
                self.word("[");
                self.commasep(Inconsistent, elts, |s, p| s.print_pat(p));
                self.word("]");
            }
            PatKind::Rest => self.word(".."),
            PatKind::Paren(inner) => {
                self.popen();
                self.print_pat(inner);
                self.pclose();
            }
            PatKind::MacCall(m) => self.print_mac(m),
        }
        self.ann.post(self, AnnNode::Pat(pat))
    }

    fn print_explicit_self(&mut self, explicit_self: &ast::ExplicitSelf) {
        match &explicit_self.node {
            SelfKind::Value(m) => {
                self.print_mutability(*m, false);
                self.word("self")
            }
            SelfKind::Region(lt, m) => {
                self.word("&");
                self.print_opt_lifetime(lt);
                self.print_mutability(*m, false);
                self.word("self")
            }
            SelfKind::Explicit(typ, m) => {
                self.print_mutability(*m, false);
                self.word("self");
                self.word_space(":");
                self.print_type(typ)
            }
        }
    }

    pub(crate) fn print_asyncness(&mut self, asyncness: ast::Async) {
        if asyncness.is_async() {
            self.word_nbsp("async");
        }
    }

    pub fn print_type_bounds(&mut self, bounds: &[ast::GenericBound]) {
        let mut first = true;
        for bound in bounds {
            if first {
                first = false;
            } else {
                self.nbsp();
                self.word_space("+");
            }

            match bound {
                GenericBound::Trait(tref, modifier) => {
                    match modifier {
                        TraitBoundModifier::None => {}
                        TraitBoundModifier::Maybe => {
                            self.word("?");
                        }
                        TraitBoundModifier::MaybeConst => {
                            self.word_space("~const");
                        }
                        TraitBoundModifier::MaybeConstMaybe => {
                            self.word_space("~const");
                            self.word("?");
                        }
                    }
                    self.print_poly_trait_ref(tref);
                }
                GenericBound::Outlives(lt) => self.print_lifetime(*lt),
            }
        }
    }

    pub(crate) fn print_lifetime(&mut self, lifetime: ast::Lifetime) {
        self.print_name(lifetime.ident.name)
    }

    pub(crate) fn print_lifetime_bounds(&mut self, bounds: &ast::GenericBounds) {
        for (i, bound) in bounds.iter().enumerate() {
            if i != 0 {
                self.word(" + ");
            }
            match bound {
                ast::GenericBound::Outlives(lt) => self.print_lifetime(*lt),
                _ => panic!(),
            }
        }
    }

    pub(crate) fn print_generic_params(&mut self, generic_params: &[ast::GenericParam]) {
        if generic_params.is_empty() {
            return;
        }

        self.word("<");

        self.commasep(Inconsistent, generic_params, |s, param| {
            s.print_outer_attributes_inline(&param.attrs);

            match &param.kind {
                ast::GenericParamKind::Lifetime => {
                    let lt = ast::Lifetime { id: param.id, ident: param.ident };
                    s.print_lifetime(lt);
                    if !param.bounds.is_empty() {
                        s.word_nbsp(":");
                        s.print_lifetime_bounds(&param.bounds)
                    }
                }
                ast::GenericParamKind::Type { default } => {
                    s.print_ident(param.ident);
                    if !param.bounds.is_empty() {
                        s.word_nbsp(":");
                        s.print_type_bounds(&param.bounds);
                    }
                    if let Some(default) = default {
                        s.space();
                        s.word_space("=");
                        s.print_type(default)
                    }
                }
                ast::GenericParamKind::Const { ty, default, .. } => {
                    s.word_space("const");
                    s.print_ident(param.ident);
                    s.space();
                    s.word_space(":");
                    s.print_type(ty);
                    if !param.bounds.is_empty() {
                        s.word_nbsp(":");
                        s.print_type_bounds(&param.bounds);
                    }
                    if let Some(default) = default {
                        s.space();
                        s.word_space("=");
                        s.print_expr(&default.value);
                    }
                }
            }
        });

        self.word(">");
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

    pub(crate) fn print_mt(&mut self, mt: &ast::MutTy, print_const: bool) {
        self.print_mutability(mt.mutbl, print_const);
        self.print_type(&mt.ty)
    }

    pub(crate) fn print_param(&mut self, input: &ast::Param, is_closure: bool) {
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
                        self.word(":");
                        self.space();
                    }
                    self.print_type(&input.ty);
                }
            }
        }
        self.end();
    }

    pub(crate) fn print_fn_ret_ty(&mut self, fn_ret_ty: &ast::FnRetTy) {
        if let ast::FnRetTy::Ty(ty) = fn_ret_ty {
            self.space_if_not_bol();
            self.ibox(INDENT_UNIT);
            self.word_space("->");
            self.print_type(ty);
            self.end();
            self.maybe_print_comment(ty.span.lo());
        }
    }

    pub(crate) fn print_ty_fn(
        &mut self,
        ext: ast::Extern,
        unsafety: ast::Unsafe,
        decl: &ast::FnDecl,
        name: Option<Ident>,
        generic_params: &[ast::GenericParam],
    ) {
        self.ibox(INDENT_UNIT);
        self.print_formal_generic_params(generic_params);
        let generics = ast::Generics {
            params: Vec::new(),
            where_clause: ast::WhereClause {
                has_where_token: false,
                predicates: Vec::new(),
                span: DUMMY_SP,
            },
            span: DUMMY_SP,
        };
        let header = ast::FnHeader { unsafety, ext, ..ast::FnHeader::default() };
        self.print_fn(decl, header, name, &generics);
        self.end();
    }

    pub(crate) fn print_fn_header_info(&mut self, header: ast::FnHeader) {
        self.print_constness(header.constness);
        self.print_asyncness(header.asyncness);
        self.print_unsafety(header.unsafety);

        match header.ext {
            ast::Extern::None => {}
            ast::Extern::Implicit(_) => {
                self.word_nbsp("extern");
            }
            ast::Extern::Explicit(abi, _) => {
                self.word_nbsp("extern");
                self.print_token_literal(abi.as_token_lit(), abi.span);
                self.nbsp();
            }
        }

        self.word("fn")
    }

    pub(crate) fn print_unsafety(&mut self, s: ast::Unsafe) {
        match s {
            ast::Unsafe::No => {}
            ast::Unsafe::Yes(_) => self.word_nbsp("unsafe"),
        }
    }

    pub(crate) fn print_constness(&mut self, s: ast::Const) {
        match s {
            ast::Const::No => {}
            ast::Const::Yes(_) => self.word_nbsp("const"),
        }
    }

    pub(crate) fn print_is_auto(&mut self, s: ast::IsAuto) {
        match s {
            ast::IsAuto::Yes => self.word_nbsp("auto"),
            ast::IsAuto::No => {}
        }
    }
}
