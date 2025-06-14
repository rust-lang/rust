//! AST pretty printing.
//!
//! Note that HIR pretty printing is layered on top of this crate.

mod expr;
mod fixup;
mod item;

use std::borrow::Cow;
use std::sync::Arc;

use rustc_ast::attr::AttrIdGenerator;
use rustc_ast::ptr::P;
use rustc_ast::token::{self, CommentKind, Delimiter, IdentIsRaw, Token, TokenKind};
use rustc_ast::tokenstream::{Spacing, TokenStream, TokenTree};
use rustc_ast::util::classify;
use rustc_ast::util::comments::{Comment, CommentStyle};
use rustc_ast::{
    self as ast, AttrArgs, BindingMode, BlockCheckMode, ByRef, DelimArgs, GenericArg, GenericBound,
    InlineAsmOperand, InlineAsmOptions, InlineAsmRegOrRegClass, InlineAsmTemplatePiece, PatKind,
    RangeEnd, RangeSyntax, Safety, SelfKind, Term, attr,
};
use rustc_span::edition::Edition;
use rustc_span::source_map::{SourceMap, Spanned};
use rustc_span::symbol::IdentPrinter;
use rustc_span::{BytePos, CharPos, DUMMY_SP, FileName, Ident, Pos, Span, Symbol, kw, sym};

use crate::pp::Breaks::{Consistent, Inconsistent};
use crate::pp::{self, BoxMarker, Breaks};
use crate::pprust::state::fixup::FixupContext;

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

struct NoAnn;

impl PpAnn for NoAnn {}

pub struct Comments<'a> {
    sm: &'a SourceMap,
    // Stored in reverse order so we can consume them by popping.
    reversed_comments: Vec<Comment>,
}

/// Returns `None` if the first `col` chars of `s` contain a non-whitespace char.
/// Otherwise returns `Some(k)` where `k` is first char offset after that leading
/// whitespace. Note that `k` may be outside bounds of `s`.
fn all_whitespace(s: &str, col: CharPos) -> Option<usize> {
    let mut idx = 0;
    for (i, ch) in s.char_indices().take(col.to_usize()) {
        if !ch.is_whitespace() {
            return None;
        }
        idx = i + ch.len_utf8();
    }
    Some(idx)
}

fn trim_whitespace_prefix(s: &str, col: CharPos) -> &str {
    let len = s.len();
    match all_whitespace(s, col) {
        Some(col) => {
            if col < len {
                &s[col..]
            } else {
                ""
            }
        }
        None => s,
    }
}

fn split_block_comment_into_lines(text: &str, col: CharPos) -> Vec<String> {
    let mut res: Vec<String> = vec![];
    let mut lines = text.lines();
    // just push the first line
    res.extend(lines.next().map(|it| it.to_string()));
    // for other lines, strip common whitespace prefix
    for line in lines {
        res.push(trim_whitespace_prefix(line, col).to_string())
    }
    res
}

fn gather_comments(sm: &SourceMap, path: FileName, src: String) -> Vec<Comment> {
    let sm = SourceMap::new(sm.path_mapping().clone());
    let source_file = sm.new_source_file(path, src);
    let text = Arc::clone(&(*source_file.src.as_ref().unwrap()));

    let text: &str = text.as_str();
    let start_bpos = source_file.start_pos;
    let mut pos = 0;
    let mut comments: Vec<Comment> = Vec::new();
    let mut code_to_the_left = false;

    if let Some(shebang_len) = rustc_lexer::strip_shebang(text) {
        comments.push(Comment {
            style: CommentStyle::Isolated,
            lines: vec![text[..shebang_len].to_string()],
            pos: start_bpos,
        });
        pos += shebang_len;
    }

    for token in rustc_lexer::tokenize(&text[pos..]) {
        let token_text = &text[pos..pos + token.len as usize];
        match token.kind {
            rustc_lexer::TokenKind::Whitespace => {
                if let Some(mut idx) = token_text.find('\n') {
                    code_to_the_left = false;
                    while let Some(next_newline) = &token_text[idx + 1..].find('\n') {
                        idx += 1 + next_newline;
                        comments.push(Comment {
                            style: CommentStyle::BlankLine,
                            lines: vec![],
                            pos: start_bpos + BytePos((pos + idx) as u32),
                        });
                    }
                }
            }
            rustc_lexer::TokenKind::BlockComment { doc_style, .. } => {
                if doc_style.is_none() {
                    let code_to_the_right = !matches!(
                        text[pos + token.len as usize..].chars().next(),
                        Some('\r' | '\n')
                    );
                    let style = match (code_to_the_left, code_to_the_right) {
                        (_, true) => CommentStyle::Mixed,
                        (false, false) => CommentStyle::Isolated,
                        (true, false) => CommentStyle::Trailing,
                    };

                    // Count the number of chars since the start of the line by rescanning.
                    let pos_in_file = start_bpos + BytePos(pos as u32);
                    let line_begin_in_file = source_file.line_begin_pos(pos_in_file);
                    let line_begin_pos = (line_begin_in_file - start_bpos).to_usize();
                    let col = CharPos(text[line_begin_pos..pos].chars().count());

                    let lines = split_block_comment_into_lines(token_text, col);
                    comments.push(Comment { style, lines, pos: pos_in_file })
                }
            }
            rustc_lexer::TokenKind::LineComment { doc_style } => {
                if doc_style.is_none() {
                    comments.push(Comment {
                        style: if code_to_the_left {
                            CommentStyle::Trailing
                        } else {
                            CommentStyle::Isolated
                        },
                        lines: vec![token_text.to_string()],
                        pos: start_bpos + BytePos(pos as u32),
                    })
                }
            }
            _ => {
                code_to_the_left = true;
            }
        }
        pos += token.len as usize;
    }

    comments
}

impl<'a> Comments<'a> {
    pub fn new(sm: &'a SourceMap, filename: FileName, input: String) -> Comments<'a> {
        let mut comments = gather_comments(sm, filename, input);
        comments.reverse();
        Comments { sm, reversed_comments: comments }
    }

    fn peek(&self) -> Option<&Comment> {
        self.reversed_comments.last()
    }

    fn next(&mut self) -> Option<Comment> {
        self.reversed_comments.pop()
    }

    fn trailing_comment(
        &mut self,
        span: rustc_span::Span,
        next_pos: Option<BytePos>,
    ) -> Option<Comment> {
        if let Some(cmnt) = self.peek() {
            if cmnt.style != CommentStyle::Trailing {
                return None;
            }
            let span_line = self.sm.lookup_char_pos(span.hi());
            let comment_line = self.sm.lookup_char_pos(cmnt.pos);
            let next = next_pos.unwrap_or_else(|| cmnt.pos + BytePos(1));
            if span.hi() < cmnt.pos && cmnt.pos < next && span_line.line == comment_line.line {
                return Some(self.next().unwrap());
            }
        }

        None
    }
}

pub struct State<'a> {
    pub s: pp::Printer,
    comments: Option<Comments<'a>>,
    ann: &'a (dyn PpAnn + 'a),
    is_sdylib_interface: bool,
}

const INDENT_UNIT: isize = 4;

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
    let mut s = State {
        s: pp::Printer::new(),
        comments: Some(Comments::new(sm, filename, input)),
        ann,
        is_sdylib_interface: false,
    };

    print_crate_inner(&mut s, krate, is_expanded, edition, g);
    s.s.eof()
}

pub fn print_crate_as_interface(
    krate: &ast::Crate,
    edition: Edition,
    g: &AttrIdGenerator,
) -> String {
    let mut s =
        State { s: pp::Printer::new(), comments: None, ann: &NoAnn, is_sdylib_interface: true };

    print_crate_inner(&mut s, krate, false, edition, g);
    s.s.eof()
}

fn print_crate_inner<'a>(
    s: &mut State<'a>,
    krate: &ast::Crate,
    is_expanded: bool,
    edition: Edition,
    g: &AttrIdGenerator,
) {
    // We need to print shebang before anything else
    // otherwise the resulting code will not compile
    // and shebang will be useless.
    s.maybe_print_shebang();

    if is_expanded && !krate.attrs.iter().any(|attr| attr.has_name(sym::no_core)) {
        // We need to print `#![no_std]` (and its feature gate) so that
        // compiling pretty-printed source won't inject libstd again.
        // However, we don't want these attributes in the AST because
        // of the feature gate, so we fake them up here.

        // `#![feature(prelude_import)]`
        let fake_attr = attr::mk_attr_nested_word(
            g,
            ast::AttrStyle::Inner,
            Safety::Default,
            sym::feature,
            sym::prelude_import,
            DUMMY_SP,
        );
        s.print_attribute(&fake_attr);

        // Currently, in Rust 2018 we don't have `extern crate std;` at the crate
        // root, so this is not needed, and actually breaks things.
        if edition.is_rust_2015() {
            // `#![no_std]`
            let fake_attr = attr::mk_attr_word(
                g,
                ast::AttrStyle::Inner,
                Safety::Default,
                sym::no_std,
                DUMMY_SP,
            );
            s.print_attribute(&fake_attr);
        }
    }

    s.print_inner_attributes(&krate.attrs);
    for item in &krate.items {
        s.print_item(item);
    }
    s.print_remaining_comments();
    s.ann.post(s, AnnNode::Crate(krate));
}

/// Should two consecutive tokens be printed with a space between them?
///
/// Note: some old proc macros parse pretty-printed output, so changes here can
/// break old code. For example:
/// - #63896: `#[allow(unused,` must be printed rather than `#[allow(unused ,`
/// - #73345: `#[allow(unused)]` must be printed rather than `# [allow(unused)]`
///
fn space_between(tt1: &TokenTree, tt2: &TokenTree) -> bool {
    use Delimiter::*;
    use TokenTree::{Delimited as Del, Token as Tok};
    use token::*;

    fn is_punct(tt: &TokenTree) -> bool {
        matches!(tt, TokenTree::Token(tok, _) if tok.is_punct())
    }

    // Each match arm has one or more examples in comments. The default is to
    // insert space between adjacent tokens, except for the cases listed in
    // this match.
    match (tt1, tt2) {
        // No space after line doc comments.
        (Tok(Token { kind: DocComment(CommentKind::Line, ..), .. }, _), _) => false,

        // `.` + NON-PUNCT: `x.y`, `tup.0`
        (Tok(Token { kind: Dot, .. }, _), tt2) if !is_punct(tt2) => false,

        // `$` + IDENT: `$e`
        (Tok(Token { kind: Dollar, .. }, _), Tok(Token { kind: Ident(..), .. }, _)) => false,

        // NON-PUNCT + `,`: `foo,`
        // NON-PUNCT + `;`: `x = 3;`, `[T; 3]`
        // NON-PUNCT + `.`: `x.y`, `tup.0`
        (tt1, Tok(Token { kind: Comma | Semi | Dot, .. }, _)) if !is_punct(tt1) => false,

        // IDENT + `!`: `println!()`, but `if !x { ... }` needs a space after the `if`
        (Tok(Token { kind: Ident(sym, is_raw), span }, _), Tok(Token { kind: Bang, .. }, _))
            if !Ident::new(*sym, *span).is_reserved() || matches!(is_raw, IdentIsRaw::Yes) =>
        {
            false
        }

        // IDENT|`fn`|`Self`|`pub` + `(`: `f(3)`, `fn(x: u8)`, `Self()`, `pub(crate)`,
        //      but `let (a, b) = (1, 2)` needs a space after the `let`
        (Tok(Token { kind: Ident(sym, is_raw), span }, _), Del(_, _, Parenthesis, _))
            if !Ident::new(*sym, *span).is_reserved()
                || *sym == kw::Fn
                || *sym == kw::SelfUpper
                || *sym == kw::Pub
                || matches!(is_raw, IdentIsRaw::Yes) =>
        {
            false
        }

        // `#` + `[`: `#[attr]`
        (Tok(Token { kind: Pound, .. }, _), Del(_, _, Bracket, _)) => false,

        _ => true,
    }
}

pub fn doc_comment_to_string(
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

fn literal_to_string(lit: token::Lit) -> String {
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
        token::CStr => format!("c\"{symbol}\""),
        token::CStrRaw(n) => {
            format!("cr{delim}\"{symbol}\"{delim}", delim = "#".repeat(n as usize))
        }
        token::Integer | token::Float | token::Bool | token::Err(_) => symbol.to_string(),
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

/// This trait is used for both AST and HIR pretty-printing.
pub trait PrintState<'a>: std::ops::Deref<Target = pp::Printer> + std::ops::DerefMut {
    fn comments(&self) -> Option<&Comments<'a>>;
    fn comments_mut(&mut self) -> Option<&mut Comments<'a>>;
    fn ann_post(&mut self, ident: Ident);
    fn print_generic_args(&mut self, args: &ast::GenericArgs, colons_before_params: bool);

    fn print_ident(&mut self, ident: Ident) {
        self.word(IdentPrinter::for_ast_ident(ident, ident.is_raw_guess()).to_string());
        self.ann_post(ident)
    }

    fn strsep<'x, T: 'x, F, I>(
        &mut self,
        sep: &'static str,
        space_before: bool,
        b: Breaks,
        elts: I,
        mut op: F,
    ) where
        F: FnMut(&mut Self, &T),
        I: IntoIterator<Item = &'x T>,
    {
        let mut it = elts.into_iter();

        let rb = self.rbox(0, b);
        if let Some(first) = it.next() {
            op(self, first);
            for elt in it {
                if space_before {
                    self.space();
                }
                self.word_space(sep);
                op(self, elt);
            }
        }
        self.end(rb);
    }

    fn commasep<'x, T: 'x, F, I>(&mut self, b: Breaks, elts: I, op: F)
    where
        F: FnMut(&mut Self, &T),
        I: IntoIterator<Item = &'x T>,
    {
        self.strsep(",", false, b, elts, op)
    }

    fn maybe_print_comment(&mut self, pos: BytePos) -> bool {
        let mut has_comment = false;
        while let Some(cmnt) = self.peek_comment() {
            if cmnt.pos >= pos {
                break;
            }
            has_comment = true;
            let cmnt = self.next_comment().unwrap();
            self.print_comment(cmnt);
        }
        has_comment
    }

    fn print_comment(&mut self, cmnt: Comment) {
        match cmnt.style {
            CommentStyle::Mixed => {
                if !self.is_beginning_of_line() {
                    self.zerobreak();
                }
                if let Some((last, lines)) = cmnt.lines.split_last() {
                    let ib = self.ibox(0);

                    for line in lines {
                        self.word(line.clone());
                        self.hardbreak()
                    }

                    self.word(last.clone());
                    self.space();

                    self.end(ib);
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
                if let [line] = cmnt.lines.as_slice() {
                    self.word(line.clone());
                    self.hardbreak()
                } else {
                    let vb = self.visual_align();
                    for line in &cmnt.lines {
                        if !line.is_empty() {
                            self.word(line.clone());
                        }
                        self.hardbreak();
                    }
                    self.end(vb);
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
    }

    fn peek_comment<'b>(&'b self) -> Option<&'b Comment>
    where
        'a: 'b,
    {
        self.comments().and_then(|c| c.peek())
    }

    fn next_comment(&mut self) -> Option<Comment> {
        self.comments_mut().and_then(|c| c.next())
    }

    fn maybe_print_trailing_comment(&mut self, span: rustc_span::Span, next_pos: Option<BytePos>) {
        if let Some(cmnts) = self.comments_mut() {
            if let Some(cmnt) = cmnts.trailing_comment(span, next_pos) {
                self.print_comment(cmnt);
            }
        }
    }

    fn print_remaining_comments(&mut self) {
        // If there aren't any remaining comments, then we need to manually
        // make sure there is a line break at the end.
        if self.peek_comment().is_none() {
            self.hardbreak();
        }
        while let Some(cmnt) = self.next_comment() {
            self.print_comment(cmnt)
        }
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

    fn maybe_print_shebang(&mut self) {
        if let Some(cmnt) = self.peek_comment() {
            // Comment is a shebang if it's:
            // Isolated, starts with #! and doesn't continue with `[`
            // See [rustc_lexer::strip_shebang] and [gather_comments] from pprust/state.rs for details
            if cmnt.style == CommentStyle::Isolated
                && cmnt.lines.first().map_or(false, |l| l.starts_with("#!"))
            {
                let cmnt = self.next_comment().unwrap();
                self.print_comment(cmnt);
            }
        }
    }

    fn print_inner_attributes(&mut self, attrs: &[ast::Attribute]) -> bool {
        self.print_either_attributes(attrs, ast::AttrStyle::Inner, false, true)
    }

    fn print_outer_attributes(&mut self, attrs: &[ast::Attribute]) -> bool {
        self.print_either_attributes(attrs, ast::AttrStyle::Outer, false, true)
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
                if self.print_attribute_inline(attr, is_inline) {
                    if is_inline {
                        self.nbsp();
                    }
                    printed = true;
                }
            }
        }
        if printed && trailing_hardbreak && !is_inline {
            self.hardbreak_if_not_bol();
        }
        printed
    }

    fn print_attribute_inline(&mut self, attr: &ast::Attribute, is_inline: bool) -> bool {
        if attr.has_name(sym::cfg_trace) || attr.has_name(sym::cfg_attr_trace) {
            // It's not a valid identifier, so avoid printing it
            // to keep the printed code reasonably parse-able.
            return false;
        }
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
        true
    }

    fn print_attr_item(&mut self, item: &ast::AttrItem, span: Span) {
        let ib = self.ibox(0);
        match item.unsafety {
            ast::Safety::Unsafe(_) => {
                self.word("unsafe");
                self.popen();
            }
            ast::Safety::Default | ast::Safety::Safe(_) => {}
        }
        match &item.args {
            AttrArgs::Delimited(DelimArgs { dspan: _, delim, tokens }) => self.print_mac_common(
                Some(MacHeader::Path(&item.path)),
                false,
                None,
                *delim,
                None,
                tokens,
                true,
                span,
            ),
            AttrArgs::Empty => {
                self.print_path(&item.path, false, 0);
            }
            AttrArgs::Eq { expr, .. } => {
                self.print_path(&item.path, false, 0);
                self.space();
                self.word_space("=");
                let token_str = self.expr_to_string(expr);
                self.word(token_str);
            }
        }
        match item.unsafety {
            ast::Safety::Unsafe(_) => self.pclose(),
            ast::Safety::Default | ast::Safety::Safe(_) => {}
        }
        self.end(ib);
    }

    /// This doesn't deserve to be called "pretty" printing, but it should be
    /// meaning-preserving. A quick hack that might help would be to look at the
    /// spans embedded in the TTs to decide where to put spaces and newlines.
    /// But it'd be better to parse these according to the grammar of the
    /// appropriate macro, transcribe back into the grammar we just parsed from,
    /// and then pretty-print the resulting AST nodes (so, e.g., we print
    /// expression arguments as expressions). It can be done! I think.
    fn print_tt(&mut self, tt: &TokenTree, convert_dollar_crate: bool) -> Spacing {
        match tt {
            TokenTree::Token(token, spacing) => {
                let token_str = self.token_to_string_ext(token, convert_dollar_crate);
                self.word(token_str);
                if let token::DocComment(..) = token.kind {
                    self.hardbreak()
                }
                *spacing
            }
            TokenTree::Delimited(dspan, spacing, delim, tts) => {
                self.print_mac_common(
                    None,
                    false,
                    None,
                    *delim,
                    Some(spacing.open),
                    tts,
                    convert_dollar_crate,
                    dspan.entire(),
                );
                spacing.close
            }
        }
    }

    // The easiest way to implement token stream pretty printing would be to
    // print each token followed by a single space. But that would produce ugly
    // output, so we go to some effort to do better.
    //
    // First, we track whether each token that appears in source code is
    // followed by a space, with `Spacing`, and reproduce that in the output.
    // This works well in a lot of cases. E.g. `stringify!(x + y)` produces
    // "x + y" and `stringify!(x+y)` produces "x+y".
    //
    // But this doesn't work for code produced by proc macros (which have no
    // original source text representation) nor for code produced by decl
    // macros (which are tricky because the whitespace after tokens appearing
    // in macro rules isn't always what you want in the produced output). For
    // these we mostly use `Spacing::Alone`, which is the conservative choice.
    //
    // So we have a backup mechanism for when `Spacing::Alone` occurs between a
    // pair of tokens: we check if that pair of tokens can obviously go
    // together without a space between them. E.g. token `x` followed by token
    // `,` is better printed as `x,` than `x ,`. (Even if the original source
    // code was `x ,`.)
    //
    // Finally, we must be careful about changing the output. Token pretty
    // printing is used by `stringify!` and `impl Display for
    // proc_macro::TokenStream`, and some programs rely on the output having a
    // particular form, even though they shouldn't. In particular, some proc
    // macros do `format!({stream})` on a token stream and then "parse" the
    // output with simple string matching that can't handle whitespace changes.
    // E.g. we have seen cases where a proc macro can handle `a :: b` but not
    // `a::b`. See #117433 for some examples.
    fn print_tts(&mut self, tts: &TokenStream, convert_dollar_crate: bool) {
        let mut iter = tts.iter().peekable();
        while let Some(tt) = iter.next() {
            let spacing = self.print_tt(tt, convert_dollar_crate);
            if let Some(next) = iter.peek() {
                if spacing == Spacing::Alone && space_between(tt, next) {
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
        open_spacing: Option<Spacing>,
        tts: &TokenStream,
        convert_dollar_crate: bool,
        span: Span,
    ) {
        let cb = (delim == Delimiter::Brace).then(|| self.cbox(INDENT_UNIT));
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

                // Respect `Alone`, if provided, and print a space. Unless the list is empty.
                let open_space = (open_spacing == None || open_spacing == Some(Spacing::Alone))
                    && !tts.is_empty();
                if open_space {
                    self.space();
                }
                let ib = self.ibox(0);
                self.print_tts(tts, convert_dollar_crate);
                self.end(ib);

                // Use `open_space` for the spacing *before* the closing delim.
                // Because spacing on delimiters is lost when going through
                // proc macros, and otherwise we can end up with ugly cases
                // like `{ x}`. Symmetry is better.
                self.bclose(span, !open_space, cb.unwrap());
            }
            delim => {
                // `open_spacing` is ignored. We never print spaces after
                // non-brace opening delims or before non-brace closing delims.
                let token_str = self.token_kind_to_string(&delim.as_open_token_kind());
                self.word(token_str);
                let ib = self.ibox(0);
                self.print_tts(tts, convert_dollar_crate);
                self.end(ib);
                let token_str = self.token_kind_to_string(&delim.as_close_token_kind());
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
            macro_def.body.delim,
            None,
            &macro_def.body.tokens,
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

    fn head<S: Into<Cow<'static, str>>>(&mut self, w: S) -> (BoxMarker, BoxMarker) {
        let w = w.into();
        // Outer-box is consistent.
        let cb = self.cbox(INDENT_UNIT);
        // Head-box is inconsistent.
        let ib = self.ibox(0);
        // Keyword that starts the head.
        if !w.is_empty() {
            self.word_nbsp(w);
        }
        (cb, ib)
    }

    fn bopen(&mut self, ib: BoxMarker) {
        self.word("{");
        self.end(ib);
    }

    fn bclose_maybe_open(&mut self, span: rustc_span::Span, no_space: bool, cb: Option<BoxMarker>) {
        let has_comment = self.maybe_print_comment(span.hi());
        if !no_space || has_comment {
            self.break_offset_if_not_bol(1, -INDENT_UNIT);
        }
        self.word("}");
        if let Some(cb) = cb {
            self.end(cb);
        }
    }

    fn bclose(&mut self, span: rustc_span::Span, no_space: bool, cb: BoxMarker) {
        let cb = Some(cb);
        self.bclose_maybe_open(span, no_space, cb)
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
            token::Bang => "!".into(),
            token::Tilde => "~".into(),
            token::OrOr => "||".into(),
            token::AndAnd => "&&".into(),
            token::Plus => "+".into(),
            token::Minus => "-".into(),
            token::Star => "*".into(),
            token::Slash => "/".into(),
            token::Percent => "%".into(),
            token::Caret => "^".into(),
            token::And => "&".into(),
            token::Or => "|".into(),
            token::Shl => "<<".into(),
            token::Shr => ">>".into(),
            token::PlusEq => "+=".into(),
            token::MinusEq => "-=".into(),
            token::StarEq => "*=".into(),
            token::SlashEq => "/=".into(),
            token::PercentEq => "%=".into(),
            token::CaretEq => "^=".into(),
            token::AndEq => "&=".into(),
            token::OrEq => "|=".into(),
            token::ShlEq => "<<=".into(),
            token::ShrEq => ">>=".into(),

            /* Structural symbols */
            token::At => "@".into(),
            token::Dot => ".".into(),
            token::DotDot => "..".into(),
            token::DotDotDot => "...".into(),
            token::DotDotEq => "..=".into(),
            token::Comma => ",".into(),
            token::Semi => ";".into(),
            token::Colon => ":".into(),
            token::PathSep => "::".into(),
            token::RArrow => "->".into(),
            token::LArrow => "<-".into(),
            token::FatArrow => "=>".into(),
            token::OpenParen => "(".into(),
            token::CloseParen => ")".into(),
            token::OpenBracket => "[".into(),
            token::CloseBracket => "]".into(),
            token::OpenBrace => "{".into(),
            token::CloseBrace => "}".into(),
            token::OpenInvisible(_) | token::CloseInvisible(_) => "".into(),
            token::Pound => "#".into(),
            token::Dollar => "$".into(),
            token::Question => "?".into(),
            token::SingleQuote => "'".into(),

            /* Literals */
            token::Literal(lit) => literal_to_string(lit).into(),

            /* Name components */
            token::Ident(name, is_raw) => {
                IdentPrinter::new(name, is_raw.into(), convert_dollar_crate).to_string().into()
            }
            token::NtIdent(ident, is_raw) => {
                IdentPrinter::for_ast_ident(ident, is_raw.into()).to_string().into()
            }

            token::Lifetime(name, IdentIsRaw::No)
            | token::NtLifetime(Ident { name, .. }, IdentIsRaw::No) => name.to_string().into(),
            token::Lifetime(name, IdentIsRaw::Yes)
            | token::NtLifetime(Ident { name, .. }, IdentIsRaw::Yes) => {
                format!("'r#{}", &name.as_str()[1..]).into()
            }

            /* Other */
            token::DocComment(comment_kind, attr_style, data) => {
                doc_comment_to_string(comment_kind, attr_style, data).into()
            }
            token::Eof => "<eof>".into(),
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

    fn pat_to_string(&self, pat: &ast::Pat) -> String {
        Self::to_string(|s| s.print_pat(pat))
    }

    fn expr_to_string(&self, e: &ast::Expr) -> String {
        Self::to_string(|s| s.print_expr(e, FixupContext::default()))
    }

    fn meta_item_lit_to_string(&self, lit: &ast::MetaItemLit) -> String {
        Self::to_string(|s| s.print_meta_item_lit(lit))
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

    fn path_to_string(&self, p: &ast::Path) -> String {
        Self::to_string(|s| s.print_path(p, false, 0))
    }

    fn vis_to_string(&self, v: &ast::Visibility) -> String {
        Self::to_string(|s| s.print_visibility(v))
    }

    fn block_to_string(&self, blk: &ast::Block) -> String {
        Self::to_string(|s| {
            let (cb, ib) = s.head("");
            s.print_block(blk, cb, ib)
        })
    }

    fn attr_item_to_string(&self, ai: &ast::AttrItem) -> String {
        Self::to_string(|s| s.print_attr_item(ai, ai.path.span))
    }

    fn tts_to_string(&self, tokens: &TokenStream) -> String {
        Self::to_string(|s| s.print_tts(tokens, false))
    }

    fn to_string(f: impl FnOnce(&mut State<'_>)) -> String {
        let mut printer = State::new();
        f(&mut printer);
        printer.s.eof()
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
        self.ann.post(self, AnnNode::Ident(&ident));
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
                    ast::AngleBracketedArg::Constraint(c) => s.print_assoc_item_constraint(c),
                });
                self.word(">")
            }

            ast::GenericArgs::Parenthesized(data) => {
                self.word("(");
                self.commasep(Inconsistent, &data.inputs, |s, ty| s.print_type(ty));
                self.word(")");
                self.print_fn_ret_ty(&data.output);
            }
            ast::GenericArgs::ParenthesizedElided(_) => {
                self.word("(");
                self.word("..");
                self.word(")");
            }
        }
    }
}

impl<'a> State<'a> {
    pub fn new() -> State<'a> {
        State { s: pp::Printer::new(), comments: None, ann: &NoAnn, is_sdylib_interface: false }
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

    fn commasep_exprs(&mut self, b: Breaks, exprs: &[P<ast::Expr>]) {
        self.commasep_cmnt(b, exprs, |s, e| s.print_expr(e, FixupContext::default()), |e| e.span)
    }

    pub fn print_opt_lifetime(&mut self, lifetime: &Option<ast::Lifetime>) {
        if let Some(lt) = *lifetime {
            self.print_lifetime(lt);
            self.nbsp();
        }
    }

    pub fn print_assoc_item_constraint(&mut self, constraint: &ast::AssocItemConstraint) {
        self.print_ident(constraint.ident);
        if let Some(args) = constraint.gen_args.as_ref() {
            self.print_generic_args(args, false)
        }
        self.space();
        match &constraint.kind {
            ast::AssocItemConstraintKind::Equality { term } => {
                self.word_space("=");
                match term {
                    Term::Ty(ty) => self.print_type(ty),
                    Term::Const(c) => self.print_expr_anon_const(c, &[]),
                }
            }
            ast::AssocItemConstraintKind::Bound { bounds } => {
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
            GenericArg::Const(ct) => self.print_expr(&ct.value, FixupContext::default()),
        }
    }

    pub fn print_ty_pat(&mut self, pat: &ast::TyPat) {
        match &pat.kind {
            rustc_ast::TyPatKind::Range(start, end, include_end) => {
                if let Some(start) = start {
                    self.print_expr_anon_const(start, &[]);
                }
                self.word("..");
                if let Some(end) = end {
                    if let RangeEnd::Included(_) = include_end.node {
                        self.word("=");
                    }
                    self.print_expr_anon_const(end, &[]);
                }
            }
            rustc_ast::TyPatKind::Or(variants) => {
                let mut first = true;
                for pat in variants {
                    if first {
                        first = false
                    } else {
                        self.word(" | ");
                    }
                    self.print_ty_pat(pat);
                }
            }
            rustc_ast::TyPatKind::Err(_) => {
                self.popen();
                self.word("/*ERROR*/");
                self.pclose();
            }
        }
    }

    pub fn print_type(&mut self, ty: &ast::Ty) {
        self.maybe_print_comment(ty.span.lo());
        let ib = self.ibox(0);
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
            ast::TyKind::PinnedRef(lifetime, mt) => {
                self.word("&");
                self.print_opt_lifetime(lifetime);
                self.word("pin ");
                self.print_mt(mt, true);
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
                self.print_ty_fn(f.ext, f.safety, &f.decl, None, &f.generic_params);
            }
            ast::TyKind::UnsafeBinder(f) => {
                let ib = self.ibox(INDENT_UNIT);
                self.word("unsafe");
                self.print_generic_params(&f.generic_params);
                self.nbsp();
                self.print_type(&f.inner_ty);
                self.end(ib);
            }
            ast::TyKind::Path(None, path) => {
                self.print_path(path, false, 0);
            }
            ast::TyKind::Path(Some(qself), path) => self.print_qpath(path, qself, false),
            ast::TyKind::TraitObject(bounds, syntax) => {
                match syntax {
                    ast::TraitObjectSyntax::Dyn => self.word_nbsp("dyn"),
                    ast::TraitObjectSyntax::DynStar => self.word_nbsp("dyn*"),
                    ast::TraitObjectSyntax::None => {}
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
                self.print_expr(&length.value, FixupContext::default());
                self.word("]");
            }
            ast::TyKind::Typeof(e) => {
                self.word("typeof(");
                self.print_expr(&e.value, FixupContext::default());
                self.word(")");
            }
            ast::TyKind::Infer => {
                self.word("_");
            }
            ast::TyKind::Err(_) => {
                self.popen();
                self.word("/*ERROR*/");
                self.pclose();
            }
            ast::TyKind::Dummy => {
                self.popen();
                self.word("/*DUMMY*/");
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
            ast::TyKind::Pat(ty, pat) => {
                self.print_type(ty);
                self.word(" is ");
                self.print_ty_pat(pat);
            }
        }
        self.end(ib);
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

        let ast::TraitBoundModifiers { constness, asyncness, polarity } = t.modifiers;
        match constness {
            ast::BoundConstness::Never => {}
            ast::BoundConstness::Always(_) | ast::BoundConstness::Maybe(_) => {
                self.word_space(constness.as_str());
            }
        }
        match asyncness {
            ast::BoundAsyncness::Normal => {}
            ast::BoundAsyncness::Async(_) => {
                self.word_space(asyncness.as_str());
            }
        }
        match polarity {
            ast::BoundPolarity::Positive => {}
            ast::BoundPolarity::Negative(_) | ast::BoundPolarity::Maybe(_) => {
                self.word(polarity.as_str());
            }
        }

        self.print_trait_ref(&t.trait_ref)
    }

    fn print_stmt(&mut self, st: &ast::Stmt) {
        self.maybe_print_comment(st.span.lo());
        match &st.kind {
            ast::StmtKind::Let(loc) => {
                self.print_outer_attributes(&loc.attrs);
                self.space_if_not_bol();
                let ib1 = self.ibox(INDENT_UNIT);
                if loc.super_.is_some() {
                    self.word_nbsp("super");
                }
                self.word_nbsp("let");

                let ib2 = self.ibox(INDENT_UNIT);
                self.print_local_decl(loc);
                self.end(ib2);
                if let Some((init, els)) = loc.kind.init_else_opt() {
                    self.nbsp();
                    self.word_space("=");
                    self.print_expr_cond_paren(
                        init,
                        els.is_some() && classify::expr_trailing_brace(init).is_some(),
                        FixupContext::default(),
                    );
                    if let Some(els) = els {
                        let cb = self.cbox(INDENT_UNIT);
                        let ib = self.ibox(INDENT_UNIT);
                        self.word(" else ");
                        self.print_block(els, cb, ib);
                    }
                }
                self.word(";");
                self.end(ib1);
            }
            ast::StmtKind::Item(item) => self.print_item(item),
            ast::StmtKind::Expr(expr) => {
                self.space_if_not_bol();
                self.print_expr_outer_attr_style(expr, false, FixupContext::new_stmt());
                if classify::expr_requires_semi_to_be_stmt(expr) {
                    self.word(";");
                }
            }
            ast::StmtKind::Semi(expr) => {
                self.space_if_not_bol();
                self.print_expr_outer_attr_style(expr, false, FixupContext::new_stmt());
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

    fn print_block(&mut self, blk: &ast::Block, cb: BoxMarker, ib: BoxMarker) {
        self.print_block_with_attrs(blk, &[], cb, ib)
    }

    fn print_block_unclosed_indent(&mut self, blk: &ast::Block, ib: BoxMarker) {
        self.print_block_maybe_unclosed(blk, &[], None, ib)
    }

    fn print_block_with_attrs(
        &mut self,
        blk: &ast::Block,
        attrs: &[ast::Attribute],
        cb: BoxMarker,
        ib: BoxMarker,
    ) {
        self.print_block_maybe_unclosed(blk, attrs, Some(cb), ib)
    }

    fn print_block_maybe_unclosed(
        &mut self,
        blk: &ast::Block,
        attrs: &[ast::Attribute],
        cb: Option<BoxMarker>,
        ib: BoxMarker,
    ) {
        match blk.rules {
            BlockCheckMode::Unsafe(..) => self.word_space("unsafe"),
            BlockCheckMode::Default => (),
        }
        self.maybe_print_comment(blk.span.lo());
        self.ann.pre(self, AnnNode::Block(blk));
        self.bopen(ib);

        let has_attrs = self.print_inner_attributes(attrs);

        for (i, st) in blk.stmts.iter().enumerate() {
            match &st.kind {
                ast::StmtKind::Expr(expr) if i == blk.stmts.len() - 1 => {
                    self.maybe_print_comment(st.span.lo());
                    self.space_if_not_bol();
                    self.print_expr_outer_attr_style(expr, false, FixupContext::new_stmt());
                    self.maybe_print_trailing_comment(expr.span, Some(blk.span.hi()));
                }
                _ => self.print_stmt(st),
            }
        }

        let no_space = !has_attrs && blk.stmts.is_empty();
        self.bclose_maybe_open(blk.span, no_space, cb);
        self.ann.post(self, AnnNode::Block(blk))
    }

    /// Print a `let pat = expr` expression.
    ///
    /// Parentheses are inserted surrounding `expr` if a round-trip through the
    /// parser would otherwise work out the wrong way in a condition position.
    ///
    /// For example each of the following would mean the wrong thing without
    /// parentheses.
    ///
    /// ```ignore (illustrative)
    /// if let _ = (Struct {}) {}
    ///
    /// if let _ = (true && false) {}
    /// ```
    ///
    /// In a match guard, the second case still requires parens, but the first
    /// case no longer does because anything until `=>` is considered part of
    /// the match guard expression. Parsing of the expression is not terminated
    /// by `{` in that position.
    ///
    /// ```ignore (illustrative)
    /// match () {
    ///     () if let _ = Struct {} => {}
    ///     () if let _ = (true && false) => {}
    /// }
    /// ```
    fn print_let(&mut self, pat: &ast::Pat, expr: &ast::Expr, fixup: FixupContext) {
        self.word("let ");
        self.print_pat(pat);
        self.space();
        self.word_space("=");
        self.print_expr_cond_paren(
            expr,
            fixup.needs_par_as_let_scrutinee(expr),
            FixupContext::default(),
        );
    }

    fn print_mac(&mut self, m: &ast::MacCall) {
        self.print_mac_common(
            Some(MacHeader::Path(&m.path)),
            true,
            None,
            m.args.delim,
            None,
            &m.args.tokens,
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
                        s.print_expr(expr, FixupContext::default());
                    }
                    InlineAsmOperand::Out { reg, late, expr } => {
                        s.word(if *late { "lateout" } else { "out" });
                        s.popen();
                        print_reg_or_class(s, reg);
                        s.pclose();
                        s.space();
                        match expr {
                            Some(expr) => s.print_expr(expr, FixupContext::default()),
                            None => s.word("_"),
                        }
                    }
                    InlineAsmOperand::InOut { reg, late, expr } => {
                        s.word(if *late { "inlateout" } else { "inout" });
                        s.popen();
                        print_reg_or_class(s, reg);
                        s.pclose();
                        s.space();
                        s.print_expr(expr, FixupContext::default());
                    }
                    InlineAsmOperand::SplitInOut { reg, late, in_expr, out_expr } => {
                        s.word(if *late { "inlateout" } else { "inout" });
                        s.popen();
                        print_reg_or_class(s, reg);
                        s.pclose();
                        s.space();
                        s.print_expr(in_expr, FixupContext::default());
                        s.space();
                        s.word_space("=>");
                        match out_expr {
                            Some(out_expr) => s.print_expr(out_expr, FixupContext::default()),
                            None => s.word("_"),
                        }
                    }
                    InlineAsmOperand::Const { anon_const } => {
                        s.word("const");
                        s.space();
                        s.print_expr(&anon_const.value, FixupContext::default());
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
                    InlineAsmOperand::Label { block } => {
                        let (cb, ib) = s.head("label");
                        s.print_block(block, cb, ib);
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
                s.commasep(Inconsistent, &opts.human_readable_names(), |s, &opt| {
                    s.word(opt);
                });
                s.pclose();
            }
        });
        self.pclose();
    }

    fn print_local_decl(&mut self, loc: &ast::Local) {
        self.print_pat(&loc.pat);
        if let Some(ty) = &loc.ty {
            self.word_space(":");
            self.print_type(ty);
        }
    }

    fn print_name(&mut self, name: Symbol) {
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

    fn print_pat(&mut self, pat: &ast::Pat) {
        self.maybe_print_comment(pat.span.lo());
        self.ann.pre(self, AnnNode::Pat(pat));
        /* Pat isn't normalized, but the beauty of it is that it doesn't matter */
        match &pat.kind {
            PatKind::Missing => unreachable!(),
            PatKind::Wild => self.word("_"),
            PatKind::Never => self.word("!"),
            PatKind::Ident(BindingMode(by_ref, mutbl), ident, sub) => {
                if mutbl.is_mut() {
                    self.word_nbsp("mut");
                }
                if let ByRef::Yes(rmutbl) = by_ref {
                    self.word_nbsp("ref");
                    if rmutbl.is_mut() {
                        self.word_nbsp("mut");
                    }
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
                let empty = fields.is_empty() && *etc == ast::PatFieldsRest::None;
                if !empty {
                    self.space();
                }
                self.commasep_cmnt(
                    Consistent,
                    fields,
                    |s, f| {
                        let cb = s.cbox(INDENT_UNIT);
                        if !f.is_shorthand {
                            s.print_ident(f.ident);
                            s.word_nbsp(":");
                        }
                        s.print_pat(&f.pat);
                        s.end(cb);
                    },
                    |f| f.pat.span,
                );
                if let ast::PatFieldsRest::Rest | ast::PatFieldsRest::Recovered(_) = etc {
                    if !fields.is_empty() {
                        self.word_space(",");
                    }
                    self.word("..");
                    if let ast::PatFieldsRest::Recovered(_) = etc {
                        self.word("/* recovered parse error */");
                    }
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
            PatKind::Deref(inner) => {
                self.word("deref!");
                self.popen();
                self.print_pat(inner);
                self.pclose();
            }
            PatKind::Ref(inner, mutbl) => {
                self.word("&");
                if mutbl.is_mut() {
                    self.word("mut ");
                }
                if let PatKind::Ident(ast::BindingMode::MUT, ..) = inner.kind {
                    self.popen();
                    self.print_pat(inner);
                    self.pclose();
                } else {
                    self.print_pat(inner);
                }
            }
            PatKind::Expr(e) => self.print_expr(e, FixupContext::default()),
            PatKind::Range(begin, end, Spanned { node: end_kind, .. }) => {
                if let Some(e) = begin {
                    self.print_expr(e, FixupContext::default());
                }
                match end_kind {
                    RangeEnd::Included(RangeSyntax::DotDotDot) => self.word("..."),
                    RangeEnd::Included(RangeSyntax::DotDotEq) => self.word("..="),
                    RangeEnd::Excluded => self.word(".."),
                }
                if let Some(e) = end {
                    self.print_expr(e, FixupContext::default());
                }
            }
            PatKind::Guard(subpat, condition) => {
                self.popen();
                self.print_pat(subpat);
                self.space();
                self.word_space("if");
                self.print_expr(condition, FixupContext::default());
                self.pclose();
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
            PatKind::Err(_) => {
                self.popen();
                self.word("/*ERROR*/");
                self.pclose();
            }
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
            SelfKind::Pinned(lt, m) => {
                self.word("&");
                self.print_opt_lifetime(lt);
                self.word("pin ");
                self.print_mutability(*m, true);
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

    fn print_coroutine_kind(&mut self, coroutine_kind: ast::CoroutineKind) {
        match coroutine_kind {
            ast::CoroutineKind::Gen { .. } => {
                self.word_nbsp("gen");
            }
            ast::CoroutineKind::Async { .. } => {
                self.word_nbsp("async");
            }
            ast::CoroutineKind::AsyncGen { .. } => {
                self.word_nbsp("async");
                self.word_nbsp("gen");
            }
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
                GenericBound::Trait(tref) => {
                    self.print_poly_trait_ref(tref);
                }
                GenericBound::Outlives(lt) => self.print_lifetime(*lt),
                GenericBound::Use(args, _) => {
                    self.word("use");
                    self.word("<");
                    self.commasep(Inconsistent, args, |s, arg| match arg {
                        ast::PreciseCapturingArg::Arg(p, _) => s.print_path(p, false, 0),
                        ast::PreciseCapturingArg::Lifetime(lt) => s.print_lifetime(*lt),
                    });
                    self.word(">")
                }
            }
        }
    }

    fn print_lifetime(&mut self, lifetime: ast::Lifetime) {
        self.print_name(lifetime.ident.name)
    }

    fn print_lifetime_bounds(&mut self, bounds: &ast::GenericBounds) {
        for (i, bound) in bounds.iter().enumerate() {
            if i != 0 {
                self.word(" + ");
            }
            match bound {
                ast::GenericBound::Outlives(lt) => self.print_lifetime(*lt),
                _ => {
                    panic!("expected a lifetime bound, found a trait bound")
                }
            }
        }
    }

    fn print_generic_params(&mut self, generic_params: &[ast::GenericParam]) {
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
                        s.print_expr(&default.value, FixupContext::default());
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

    fn print_mt(&mut self, mt: &ast::MutTy, print_const: bool) {
        self.print_mutability(mt.mutbl, print_const);
        self.print_type(&mt.ty)
    }

    fn print_param(&mut self, input: &ast::Param, is_closure: bool) {
        let ib = self.ibox(INDENT_UNIT);

        self.print_outer_attributes_inline(&input.attrs);

        match input.ty.kind {
            ast::TyKind::Infer if is_closure => self.print_pat(&input.pat),
            _ => {
                if let Some(eself) = input.to_self() {
                    self.print_explicit_self(&eself);
                } else {
                    if !matches!(input.pat.kind, PatKind::Missing) {
                        self.print_pat(&input.pat);
                        self.word(":");
                        self.space();
                    }
                    self.print_type(&input.ty);
                }
            }
        }
        self.end(ib);
    }

    fn print_fn_ret_ty(&mut self, fn_ret_ty: &ast::FnRetTy) {
        if let ast::FnRetTy::Ty(ty) = fn_ret_ty {
            self.space_if_not_bol();
            let ib = self.ibox(INDENT_UNIT);
            self.word_space("->");
            self.print_type(ty);
            self.end(ib);
            self.maybe_print_comment(ty.span.lo());
        }
    }

    fn print_ty_fn(
        &mut self,
        ext: ast::Extern,
        safety: ast::Safety,
        decl: &ast::FnDecl,
        name: Option<Ident>,
        generic_params: &[ast::GenericParam],
    ) {
        let ib = self.ibox(INDENT_UNIT);
        self.print_formal_generic_params(generic_params);
        let generics = ast::Generics::default();
        let header = ast::FnHeader { safety, ext, ..ast::FnHeader::default() };
        self.print_fn(decl, header, name, &generics);
        self.end(ib);
    }

    fn print_fn_header_info(&mut self, header: ast::FnHeader) {
        self.print_constness(header.constness);
        header.coroutine_kind.map(|coroutine_kind| self.print_coroutine_kind(coroutine_kind));
        self.print_safety(header.safety);

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

    fn print_safety(&mut self, s: ast::Safety) {
        match s {
            ast::Safety::Default => {}
            ast::Safety::Safe(_) => self.word_nbsp("safe"),
            ast::Safety::Unsafe(_) => self.word_nbsp("unsafe"),
        }
    }

    fn print_constness(&mut self, s: ast::Const) {
        match s {
            ast::Const::No => {}
            ast::Const::Yes(_) => self.word_nbsp("const"),
        }
    }

    fn print_is_auto(&mut self, s: ast::IsAuto) {
        match s {
            ast::IsAuto::Yes => self.word_nbsp("auto"),
            ast::IsAuto::No => {}
        }
    }

    fn print_meta_item_lit(&mut self, lit: &ast::MetaItemLit) {
        self.print_token_literal(lit.as_token_lit(), lit.span)
    }

    fn print_token_literal(&mut self, token_lit: token::Lit, span: Span) {
        self.maybe_print_comment(span.lo());
        self.word(token_lit.to_string())
    }

    fn print_symbol(&mut self, sym: Symbol, style: ast::StrStyle) {
        self.print_string(sym.as_str(), style);
    }

    fn print_inner_attributes_no_trailing_hardbreak(&mut self, attrs: &[ast::Attribute]) -> bool {
        self.print_either_attributes(attrs, ast::AttrStyle::Inner, false, false)
    }

    fn print_outer_attributes_inline(&mut self, attrs: &[ast::Attribute]) -> bool {
        self.print_either_attributes(attrs, ast::AttrStyle::Outer, true, true)
    }

    fn print_attribute(&mut self, attr: &ast::Attribute) {
        self.print_attribute_inline(attr, false);
    }

    fn print_meta_list_item(&mut self, item: &ast::MetaItemInner) {
        match item {
            ast::MetaItemInner::MetaItem(mi) => self.print_meta_item(mi),
            ast::MetaItemInner::Lit(lit) => self.print_meta_item_lit(lit),
        }
    }

    fn print_meta_item(&mut self, item: &ast::MetaItem) {
        let ib = self.ibox(INDENT_UNIT);
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
        self.end(ib);
    }

    pub(crate) fn bounds_to_string(&self, bounds: &[ast::GenericBound]) -> String {
        Self::to_string(|s| s.print_type_bounds(bounds))
    }

    pub(crate) fn where_bound_predicate_to_string(
        &self,
        where_bound_predicate: &ast::WhereBoundPredicate,
    ) -> String {
        Self::to_string(|s| s.print_where_bound_predicate(where_bound_predicate))
    }

    pub(crate) fn tt_to_string(&self, tt: &TokenTree) -> String {
        Self::to_string(|s| {
            s.print_tt(tt, false);
        })
    }

    pub(crate) fn path_segment_to_string(&self, p: &ast::PathSegment) -> String {
        Self::to_string(|s| s.print_path_segment(p, false))
    }

    pub(crate) fn meta_list_item_to_string(&self, li: &ast::MetaItemInner) -> String {
        Self::to_string(|s| s.print_meta_list_item(li))
    }

    pub(crate) fn attribute_to_string(&self, attr: &ast::Attribute) -> String {
        Self::to_string(|s| s.print_attribute(attr))
    }
}
