// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Format list-like macro invocations. These are invocations whose token trees
// can be interpreted as expressions and separated by commas.
// Note that these token trees do not actually have to be interpreted as
// expressions by the compiler. An example of an invocation we would reformat is
// foo!( x, y, z ). The token x may represent an identifier in the code, but we
// interpreted as an expression.
// Macro uses which are not-list like, such as bar!(key => val), will not be
// reformatted.
// List-like invocations with parentheses will be formatted as function calls,
// and those with brackets will be formatted as array literals.

use std::collections::HashMap;

use config::lists::*;
use syntax::parse::new_parser_from_tts;
use syntax::parse::parser::Parser;
use syntax::parse::token::{BinOpToken, DelimToken, Token};
use syntax::print::pprust;
use syntax::source_map::{BytePos, Span};
use syntax::symbol;
use syntax::tokenstream::{Cursor, ThinTokenStream, TokenStream, TokenTree};
use syntax::ThinVec;
use syntax::{ast, ptr};

use comment::{
    contains_comment, remove_trailing_white_spaces, CharClasses, FindUncommented, FullCodeCharKind,
    LineClasses,
};
use expr::rewrite_array;
use lists::{itemize_list, write_list, ListFormatting};
use overflow;
use rewrite::{Rewrite, RewriteContext};
use shape::{Indent, Shape};
use source_map::SpanUtils;
use spanned::Spanned;
use utils::{format_visibility, mk_sp, rewrite_ident, wrap_str};

const FORCED_BRACKET_MACROS: &[&str] = &["vec!"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroPosition {
    Item,
    Statement,
    Expression,
    Pat,
}

#[derive(Debug)]
pub enum MacroArg {
    Expr(ptr::P<ast::Expr>),
    Ty(ptr::P<ast::Ty>),
    Pat(ptr::P<ast::Pat>),
    Item(ptr::P<ast::Item>),
}

impl Rewrite for ast::Item {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        let mut visitor = ::visitor::FmtVisitor::from_context(context);
        visitor.block_indent = shape.indent;
        visitor.last_pos = self.span().lo();
        visitor.visit_item(self);
        Some(visitor.buffer)
    }
}

impl Rewrite for MacroArg {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        match *self {
            MacroArg::Expr(ref expr) => expr.rewrite(context, shape),
            MacroArg::Ty(ref ty) => ty.rewrite(context, shape),
            MacroArg::Pat(ref pat) => pat.rewrite(context, shape),
            MacroArg::Item(ref item) => item.rewrite(context, shape),
        }
    }
}

fn parse_macro_arg(parser: &mut Parser) -> Option<MacroArg> {
    macro_rules! parse_macro_arg {
        ($macro_arg:ident, $parser:ident, $f:expr) => {
            let mut cloned_parser = (*parser).clone();
            match cloned_parser.$parser() {
                Ok(x) => {
                    if parser.sess.span_diagnostic.has_errors() {
                        parser.sess.span_diagnostic.reset_err_count();
                    } else {
                        // Parsing succeeded.
                        *parser = cloned_parser;
                        return Some(MacroArg::$macro_arg($f(x)?));
                    }
                }
                Err(mut e) => {
                    e.cancel();
                    parser.sess.span_diagnostic.reset_err_count();
                }
            }
        };
    }

    parse_macro_arg!(Expr, parse_expr, |x: ptr::P<ast::Expr>| Some(x));
    parse_macro_arg!(Ty, parse_ty, |x: ptr::P<ast::Ty>| Some(x));
    parse_macro_arg!(Pat, parse_pat, |x: ptr::P<ast::Pat>| Some(x));
    // `parse_item` returns `Option<ptr::P<ast::Item>>`.
    parse_macro_arg!(Item, parse_item, |x: Option<ptr::P<ast::Item>>| x);

    None
}

/// Rewrite macro name without using pretty-printer if possible.
fn rewrite_macro_name(
    context: &RewriteContext,
    path: &ast::Path,
    extra_ident: Option<ast::Ident>,
) -> String {
    let name = if path.segments.len() == 1 {
        // Avoid using pretty-printer in the common case.
        format!("{}!", rewrite_ident(context, path.segments[0].ident))
    } else {
        format!("{}!", path)
    };
    match extra_ident {
        Some(ident) if ident != symbol::keywords::Invalid.ident() => format!("{} {}", name, ident),
        _ => name,
    }
}

// Use this on failing to format the macro call.
fn return_original_snippet_with_failure_marked(
    context: &RewriteContext,
    span: Span,
) -> Option<String> {
    context.macro_rewrite_failure.replace(true);
    Some(context.snippet(span).to_owned())
}

struct InsideMacroGuard<'a> {
    context: &'a RewriteContext<'a>,
    is_nested: bool,
}

impl<'a> InsideMacroGuard<'a> {
    fn inside_macro_context(context: &'a RewriteContext) -> InsideMacroGuard<'a> {
        let is_nested = context.inside_macro.replace(true);
        InsideMacroGuard { context, is_nested }
    }
}

impl<'a> Drop for InsideMacroGuard<'a> {
    fn drop(&mut self) {
        self.context.inside_macro.replace(self.is_nested);
    }
}

pub fn rewrite_macro(
    mac: &ast::Mac,
    extra_ident: Option<ast::Ident>,
    context: &RewriteContext,
    shape: Shape,
    position: MacroPosition,
) -> Option<String> {
    let guard = InsideMacroGuard::inside_macro_context(context);
    let result = rewrite_macro_inner(mac, extra_ident, context, shape, position, guard.is_nested);
    if result.is_none() {
        context.macro_rewrite_failure.replace(true);
    }
    result
}

pub fn rewrite_macro_inner(
    mac: &ast::Mac,
    extra_ident: Option<ast::Ident>,
    context: &RewriteContext,
    shape: Shape,
    position: MacroPosition,
    is_nested_macro: bool,
) -> Option<String> {
    if context.config.use_try_shorthand() {
        if let Some(expr) = convert_try_mac(mac, context) {
            context.inside_macro.replace(false);
            return expr.rewrite(context, shape);
        }
    }

    let original_style = macro_style(mac, context);

    let macro_name = rewrite_macro_name(context, &mac.node.path, extra_ident);

    let style = if FORCED_BRACKET_MACROS.contains(&&macro_name[..]) && !is_nested_macro {
        DelimToken::Bracket
    } else {
        original_style
    };

    let ts: TokenStream = mac.node.stream();
    let has_comment = contains_comment(context.snippet(mac.span));
    if ts.is_empty() && !has_comment {
        return match style {
            DelimToken::Paren if position == MacroPosition::Item => {
                Some(format!("{}();", macro_name))
            }
            DelimToken::Paren => Some(format!("{}()", macro_name)),
            DelimToken::Bracket => Some(format!("{}[]", macro_name)),
            DelimToken::Brace => Some(format!("{}{{}}", macro_name)),
            _ => unreachable!(),
        };
    }
    // Format well-known macros which cannot be parsed as a valid AST.
    if macro_name == "lazy_static!" && !has_comment {
        if let success @ Some(..) = format_lazy_static(context, shape, &ts) {
            return success;
        }
    }

    let mut parser = new_parser_from_tts(context.parse_session, ts.trees().collect());
    let mut arg_vec = Vec::new();
    let mut vec_with_semi = false;
    let mut trailing_comma = false;

    if DelimToken::Brace != style {
        loop {
            match parse_macro_arg(&mut parser) {
                Some(arg) => arg_vec.push(arg),
                None => return return_original_snippet_with_failure_marked(context, mac.span),
            }

            match parser.token {
                Token::Eof => break,
                Token::Comma => (),
                Token::Semi => {
                    // Try to parse `vec![expr; expr]`
                    if FORCED_BRACKET_MACROS.contains(&&macro_name[..]) {
                        parser.bump();
                        if parser.token != Token::Eof {
                            match parse_macro_arg(&mut parser) {
                                Some(arg) => {
                                    arg_vec.push(arg);
                                    parser.bump();
                                    if parser.token == Token::Eof && arg_vec.len() == 2 {
                                        vec_with_semi = true;
                                        break;
                                    }
                                }
                                None => {
                                    return return_original_snippet_with_failure_marked(
                                        context, mac.span,
                                    )
                                }
                            }
                        }
                    }
                    return return_original_snippet_with_failure_marked(context, mac.span);
                }
                _ => return return_original_snippet_with_failure_marked(context, mac.span),
            }

            parser.bump();

            if parser.token == Token::Eof {
                trailing_comma = true;
                break;
            }
        }
    }

    match style {
        DelimToken::Paren => {
            // Format macro invocation as function call, preserve the trailing
            // comma because not all macros support them.
            overflow::rewrite_with_parens(
                context,
                &macro_name,
                &arg_vec.iter().map(|e| &*e).collect::<Vec<_>>(),
                shape,
                mac.span,
                context.config.width_heuristics().fn_call_width,
                if trailing_comma {
                    Some(SeparatorTactic::Always)
                } else {
                    Some(SeparatorTactic::Never)
                },
            ).map(|rw| match position {
                MacroPosition::Item => format!("{};", rw),
                _ => rw,
            })
        }
        DelimToken::Bracket => {
            // Handle special case: `vec![expr; expr]`
            if vec_with_semi {
                let mac_shape = shape.offset_left(macro_name.len())?;
                // 8 = `vec![]` + `; `
                let total_overhead = 8;
                let nested_shape = mac_shape.block_indent(context.config.tab_spaces());
                let lhs = arg_vec[0].rewrite(context, nested_shape)?;
                let rhs = arg_vec[1].rewrite(context, nested_shape)?;
                if !lhs.contains('\n')
                    && !rhs.contains('\n')
                    && lhs.len() + rhs.len() + total_overhead <= shape.width
                {
                    Some(format!("{}[{}; {}]", macro_name, lhs, rhs))
                } else {
                    Some(format!(
                        "{}[{}{};{}{}{}]",
                        macro_name,
                        nested_shape.indent.to_string_with_newline(context.config),
                        lhs,
                        nested_shape.indent.to_string_with_newline(context.config),
                        rhs,
                        shape.indent.to_string_with_newline(context.config),
                    ))
                }
            } else {
                // If we are rewriting `vec!` macro or other special macros,
                // then we can rewrite this as an usual array literal.
                // Otherwise, we must preserve the original existence of trailing comma.
                let macro_name = &macro_name.as_str();
                let mut force_trailing_comma = if trailing_comma {
                    Some(SeparatorTactic::Always)
                } else {
                    Some(SeparatorTactic::Never)
                };
                if FORCED_BRACKET_MACROS.contains(macro_name) && !is_nested_macro {
                    context.inside_macro.replace(false);
                    if context.use_block_indent() {
                        force_trailing_comma = Some(SeparatorTactic::Vertical);
                    };
                }
                // Convert `MacroArg` into `ast::Expr`, as `rewrite_array` only accepts the latter.
                let arg_vec = &arg_vec.iter().map(|e| &*e).collect::<Vec<_>>();
                let rewrite = rewrite_array(
                    macro_name,
                    arg_vec,
                    mac.span,
                    context,
                    shape,
                    force_trailing_comma,
                    Some(original_style),
                )?;
                let comma = match position {
                    MacroPosition::Item => ";",
                    _ => "",
                };

                Some(format!("{}{}", rewrite, comma))
            }
        }
        DelimToken::Brace => {
            // Skip macro invocations with braces, for now.
            indent_macro_snippet(context, context.snippet(mac.span), shape.indent)
        }
        _ => unreachable!(),
    }
}

pub fn rewrite_macro_def(
    context: &RewriteContext,
    shape: Shape,
    indent: Indent,
    def: &ast::MacroDef,
    ident: ast::Ident,
    vis: &ast::Visibility,
    span: Span,
) -> Option<String> {
    let snippet = Some(remove_trailing_white_spaces(context.snippet(span)));
    if snippet.as_ref().map_or(true, |s| s.ends_with(';')) {
        return snippet;
    }

    let mut parser = MacroParser::new(def.stream().into_trees());
    let parsed_def = match parser.parse() {
        Some(def) => def,
        None => return snippet,
    };

    let mut result = if def.legacy {
        String::from("macro_rules!")
    } else {
        format!("{}macro", format_visibility(context, vis))
    };

    result += " ";
    result += rewrite_ident(context, ident);

    let multi_branch_style = def.legacy || parsed_def.branches.len() != 1;

    let arm_shape = if multi_branch_style {
        shape
            .block_indent(context.config.tab_spaces())
            .with_max_width(context.config)
    } else {
        shape
    };

    let branch_items = itemize_list(
        context.snippet_provider,
        parsed_def.branches.iter(),
        "}",
        ";",
        |branch| branch.span.lo(),
        |branch| branch.span.hi(),
        |branch| branch.rewrite(context, arm_shape, multi_branch_style),
        context.snippet_provider.span_after(span, "{"),
        span.hi(),
        false,
    ).collect::<Vec<_>>();

    let fmt = ListFormatting::new(arm_shape, context.config)
        .separator(if def.legacy { ";" } else { "" })
        .trailing_separator(SeparatorTactic::Always)
        .preserve_newline(true);

    if multi_branch_style {
        result += " {";
        result += &arm_shape.indent.to_string_with_newline(context.config);
    }

    match write_list(&branch_items, &fmt) {
        Some(ref s) => result += s,
        None => return snippet,
    }

    if multi_branch_style {
        result += &indent.to_string_with_newline(context.config);
        result += "}";
    }

    Some(result)
}

fn register_metavariable(
    map: &mut HashMap<String, String>,
    result: &mut String,
    name: &str,
    dollar_count: usize,
) {
    let mut new_name = String::new();
    let mut old_name = String::new();

    old_name.push('$');
    for _ in 0..(dollar_count - 1) {
        new_name.push('$');
        old_name.push('$');
    }
    new_name.push('z');
    new_name.push_str(&name);
    old_name.push_str(&name);

    result.push_str(&new_name);
    map.insert(old_name, new_name);
}

// Replaces `$foo` with `zfoo`. We must check for name overlap to ensure we
// aren't causing problems.
// This should also work for escaped `$` variables, where we leave earlier `$`s.
fn replace_names(input: &str) -> Option<(String, HashMap<String, String>)> {
    // Each substitution will require five or six extra bytes.
    let mut result = String::with_capacity(input.len() + 64);
    let mut substs = HashMap::new();
    let mut dollar_count = 0;
    let mut cur_name = String::new();

    for (kind, c) in CharClasses::new(input.chars()) {
        if kind != FullCodeCharKind::Normal {
            result.push(c);
        } else if c == '$' {
            dollar_count += 1;
        } else if dollar_count == 0 {
            result.push(c);
        } else if !c.is_alphanumeric() && !cur_name.is_empty() {
            // Terminates a name following one or more dollars.
            register_metavariable(&mut substs, &mut result, &cur_name, dollar_count);

            result.push(c);
            dollar_count = 0;
            cur_name.clear();
        } else if c == '(' && cur_name.is_empty() {
            // FIXME: Support macro def with repeat.
            return None;
        } else if c.is_alphanumeric() || c == '_' {
            cur_name.push(c);
        }
    }

    if !cur_name.is_empty() {
        register_metavariable(&mut substs, &mut result, &cur_name, dollar_count);
    }

    debug!("replace_names `{}` {:?}", result, substs);

    Some((result, substs))
}

#[derive(Debug, Clone)]
enum MacroArgKind {
    /// e.g. `$x: expr`.
    MetaVariable(ast::Ident, String),
    /// e.g. `$($foo: expr),*`
    Repeat(
        /// `()`, `[]` or `{}`.
        DelimToken,
        /// Inner arguments inside delimiters.
        Vec<ParsedMacroArg>,
        /// Something after the closing delimiter and the repeat token, if available.
        Option<Box<ParsedMacroArg>>,
        /// The repeat token. This could be one of `*`, `+` or `?`.
        Token,
    ),
    /// e.g. `[derive(Debug)]`
    Delimited(DelimToken, Vec<ParsedMacroArg>),
    /// A possible separator. e.g. `,` or `;`.
    Separator(String, String),
    /// Other random stuff that does not fit to other kinds.
    /// e.g. `== foo` in `($x: expr == foo)`.
    Other(String, String),
}

fn delim_token_to_str(
    context: &RewriteContext,
    delim_token: &DelimToken,
    shape: Shape,
    use_multiple_lines: bool,
    inner_is_empty: bool,
) -> (String, String) {
    let (lhs, rhs) = match *delim_token {
        DelimToken::Paren => ("(", ")"),
        DelimToken::Bracket => ("[", "]"),
        DelimToken::Brace => {
            if inner_is_empty || use_multiple_lines {
                ("{", "}")
            } else {
                ("{ ", " }")
            }
        }
        DelimToken::NoDelim => ("", ""),
    };
    if use_multiple_lines {
        let indent_str = shape.indent.to_string_with_newline(context.config);
        let nested_indent_str = shape
            .indent
            .block_indent(context.config)
            .to_string_with_newline(context.config);
        (
            format!("{}{}", lhs, nested_indent_str),
            format!("{}{}", indent_str, rhs),
        )
    } else {
        (lhs.to_owned(), rhs.to_owned())
    }
}

impl MacroArgKind {
    fn starts_with_brace(&self) -> bool {
        match *self {
            MacroArgKind::Repeat(DelimToken::Brace, _, _, _)
            | MacroArgKind::Delimited(DelimToken::Brace, _) => true,
            _ => false,
        }
    }

    fn starts_with_dollar(&self) -> bool {
        match *self {
            MacroArgKind::Repeat(..) | MacroArgKind::MetaVariable(..) => true,
            _ => false,
        }
    }

    fn ends_with_space(&self) -> bool {
        match *self {
            MacroArgKind::Separator(..) => true,
            _ => false,
        }
    }

    fn has_meta_var(&self) -> bool {
        match *self {
            MacroArgKind::MetaVariable(..) => true,
            MacroArgKind::Repeat(_, ref args, _, _) => args.iter().any(|a| a.kind.has_meta_var()),
            _ => false,
        }
    }

    fn rewrite(
        &self,
        context: &RewriteContext,
        shape: Shape,
        use_multiple_lines: bool,
    ) -> Option<String> {
        let rewrite_delimited_inner = |delim_tok, args| -> Option<(String, String, String)> {
            let inner = wrap_macro_args(context, args, shape)?;
            let (lhs, rhs) = delim_token_to_str(context, delim_tok, shape, false, inner.is_empty());
            if lhs.len() + inner.len() + rhs.len() <= shape.width {
                return Some((lhs, inner, rhs));
            }

            let (lhs, rhs) = delim_token_to_str(context, delim_tok, shape, true, false);
            let nested_shape = shape
                .block_indent(context.config.tab_spaces())
                .with_max_width(context.config);
            let inner = wrap_macro_args(context, args, nested_shape)?;
            Some((lhs, inner, rhs))
        };

        match *self {
            MacroArgKind::MetaVariable(ty, ref name) => {
                Some(format!("${}:{}", name, ty.name.as_str()))
            }
            MacroArgKind::Repeat(ref delim_tok, ref args, ref another, ref tok) => {
                let (lhs, inner, rhs) = rewrite_delimited_inner(delim_tok, args)?;
                let another = another
                    .as_ref()
                    .and_then(|a| a.rewrite(context, shape, use_multiple_lines))
                    .unwrap_or_else(|| "".to_owned());
                let repeat_tok = pprust::token_to_string(tok);

                Some(format!("${}{}{}{}{}", lhs, inner, rhs, another, repeat_tok))
            }
            MacroArgKind::Delimited(ref delim_tok, ref args) => {
                rewrite_delimited_inner(delim_tok, args)
                    .map(|(lhs, inner, rhs)| format!("{}{}{}", lhs, inner, rhs))
            }
            MacroArgKind::Separator(ref sep, ref prefix) => Some(format!("{}{} ", prefix, sep)),
            MacroArgKind::Other(ref inner, ref prefix) => Some(format!("{}{}", prefix, inner)),
        }
    }
}

#[derive(Debug, Clone)]
struct ParsedMacroArg {
    kind: MacroArgKind,
    span: Span,
}

impl ParsedMacroArg {
    pub fn rewrite(
        &self,
        context: &RewriteContext,
        shape: Shape,
        use_multiple_lines: bool,
    ) -> Option<String> {
        self.kind.rewrite(context, shape, use_multiple_lines)
    }
}

/// Parses macro arguments on macro def.
struct MacroArgParser {
    /// Holds either a name of the next metavariable, a separator or a junk.
    buf: String,
    /// The start position on the current buffer.
    lo: BytePos,
    /// The first token of the current buffer.
    start_tok: Token,
    /// Set to true if we are parsing a metavariable or a repeat.
    is_meta_var: bool,
    /// The position of the last token.
    hi: BytePos,
    /// The last token parsed.
    last_tok: Token,
    /// Holds the parsed arguments.
    result: Vec<ParsedMacroArg>,
}

fn last_tok(tt: &TokenTree) -> Token {
    match *tt {
        TokenTree::Token(_, ref t) => t.clone(),
        TokenTree::Delimited(_, ref d) => d.close_token(),
    }
}

impl MacroArgParser {
    pub fn new() -> MacroArgParser {
        MacroArgParser {
            lo: BytePos(0),
            hi: BytePos(0),
            buf: String::new(),
            is_meta_var: false,
            last_tok: Token::Eof,
            start_tok: Token::Eof,
            result: vec![],
        }
    }

    fn set_last_tok(&mut self, tok: &TokenTree) {
        self.hi = tok.span().hi();
        self.last_tok = last_tok(tok);
    }

    fn add_separator(&mut self) {
        let prefix = if self.need_space_prefix() {
            " ".to_owned()
        } else {
            "".to_owned()
        };
        self.result.push(ParsedMacroArg {
            kind: MacroArgKind::Separator(self.buf.clone(), prefix),
            span: mk_sp(self.lo, self.hi),
        });
        self.buf.clear();
    }

    fn add_other(&mut self) {
        let prefix = if self.need_space_prefix() {
            " ".to_owned()
        } else {
            "".to_owned()
        };
        self.result.push(ParsedMacroArg {
            kind: MacroArgKind::Other(self.buf.clone(), prefix),
            span: mk_sp(self.lo, self.hi),
        });
        self.buf.clear();
    }

    fn add_meta_variable(&mut self, iter: &mut Cursor) -> Option<()> {
        match iter.next() {
            Some(TokenTree::Token(sp, Token::Ident(ref ident, _))) => {
                self.result.push(ParsedMacroArg {
                    kind: MacroArgKind::MetaVariable(*ident, self.buf.clone()),
                    span: mk_sp(self.lo, sp.hi()),
                });

                self.buf.clear();
                self.is_meta_var = false;
                Some(())
            }
            _ => None,
        }
    }

    fn add_delimited(&mut self, inner: Vec<ParsedMacroArg>, delim: DelimToken, span: Span) {
        self.result.push(ParsedMacroArg {
            kind: MacroArgKind::Delimited(delim, inner),
            span,
        });
    }

    // $($foo: expr),?
    fn add_repeat(
        &mut self,
        inner: Vec<ParsedMacroArg>,
        delim: DelimToken,
        iter: &mut Cursor,
        span: Span,
    ) -> Option<()> {
        let mut buffer = String::new();
        let mut first = false;
        let mut lo = span.lo();
        let mut hi = span.hi();

        // Parse '*', '+' or '?.
        for ref tok in iter {
            self.set_last_tok(tok);
            if first {
                first = false;
                lo = tok.span().lo();
            }

            match tok {
                TokenTree::Token(_, Token::BinOp(BinOpToken::Plus))
                | TokenTree::Token(_, Token::Question)
                | TokenTree::Token(_, Token::BinOp(BinOpToken::Star)) => {
                    break;
                }
                TokenTree::Token(sp, ref t) => {
                    buffer.push_str(&pprust::token_to_string(t));
                    hi = sp.hi();
                }
                _ => return None,
            }
        }

        // There could be some random stuff between ')' and '*', '+' or '?'.
        let another = if buffer.trim().is_empty() {
            None
        } else {
            Some(Box::new(ParsedMacroArg {
                kind: MacroArgKind::Other(buffer, "".to_owned()),
                span: mk_sp(lo, hi),
            }))
        };

        self.result.push(ParsedMacroArg {
            kind: MacroArgKind::Repeat(delim, inner, another, self.last_tok.clone()),
            span: mk_sp(self.lo, self.hi),
        });
        Some(())
    }

    fn update_buffer(&mut self, lo: BytePos, t: &Token) {
        if self.buf.is_empty() {
            self.lo = lo;
            self.start_tok = t.clone();
        } else {
            let needs_space = match next_space(&self.last_tok) {
                SpaceState::Ident => ident_like(t),
                SpaceState::Punctuation => !ident_like(t),
                SpaceState::Always => true,
                SpaceState::Never => false,
            };
            if force_space_before(t) || needs_space {
                self.buf.push(' ');
            }
        }

        self.buf.push_str(&pprust::token_to_string(t));
    }

    fn need_space_prefix(&self) -> bool {
        if self.result.is_empty() {
            return false;
        }

        let last_arg = self.result.last().unwrap();
        if let MacroArgKind::MetaVariable(..) = last_arg.kind {
            if ident_like(&self.start_tok) {
                return true;
            }
            if self.start_tok == Token::Colon {
                return true;
            }
        }

        if force_space_before(&self.start_tok) {
            return true;
        }

        false
    }

    /// Returns a collection of parsed macro def's arguments.
    pub fn parse(mut self, tokens: ThinTokenStream) -> Option<Vec<ParsedMacroArg>> {
        let mut iter = (tokens.into(): TokenStream).trees();

        while let Some(ref tok) = iter.next() {
            match tok {
                TokenTree::Token(sp, Token::Dollar) => {
                    // We always want to add a separator before meta variables.
                    if !self.buf.is_empty() {
                        self.add_separator();
                    }

                    // Start keeping the name of this metavariable in the buffer.
                    self.is_meta_var = true;
                    self.lo = sp.lo();
                    self.start_tok = Token::Dollar;
                }
                TokenTree::Token(_, Token::Colon) if self.is_meta_var => {
                    self.add_meta_variable(&mut iter)?;
                }
                TokenTree::Token(sp, ref t) => self.update_buffer(sp.lo(), t),
                TokenTree::Delimited(sp, delimited) => {
                    if !self.buf.is_empty() {
                        if next_space(&self.last_tok) == SpaceState::Always {
                            self.add_separator();
                        } else {
                            self.add_other();
                        }
                    }

                    // Parse the stuff inside delimiters.
                    let mut parser = MacroArgParser::new();
                    parser.lo = sp.lo();
                    let delimited_arg = parser.parse(delimited.tts.clone())?;

                    if self.is_meta_var {
                        self.add_repeat(delimited_arg, delimited.delim, &mut iter, *sp)?;
                        self.is_meta_var = false;
                    } else {
                        self.add_delimited(delimited_arg, delimited.delim, *sp);
                    }
                }
            }

            self.set_last_tok(tok);
        }

        // We are left with some stuff in the buffer. Since there is nothing
        // left to separate, add this as `Other`.
        if !self.buf.is_empty() {
            self.add_other();
        }

        Some(self.result)
    }
}

fn wrap_macro_args(
    context: &RewriteContext,
    args: &[ParsedMacroArg],
    shape: Shape,
) -> Option<String> {
    wrap_macro_args_inner(context, args, shape, false)
        .or_else(|| wrap_macro_args_inner(context, args, shape, true))
}

fn wrap_macro_args_inner(
    context: &RewriteContext,
    args: &[ParsedMacroArg],
    shape: Shape,
    use_multiple_lines: bool,
) -> Option<String> {
    let mut result = String::with_capacity(128);
    let mut iter = args.iter().peekable();
    let indent_str = shape.indent.to_string_with_newline(context.config);

    while let Some(ref arg) = iter.next() {
        result.push_str(&arg.rewrite(context, shape, use_multiple_lines)?);

        if use_multiple_lines
            && (arg.kind.ends_with_space() || iter.peek().map_or(false, |a| a.kind.has_meta_var()))
        {
            if arg.kind.ends_with_space() {
                result.pop();
            }
            result.push_str(&indent_str);
        } else if let Some(ref next_arg) = iter.peek() {
            let space_before_dollar =
                !arg.kind.ends_with_space() && next_arg.kind.starts_with_dollar();
            let space_before_brace = next_arg.kind.starts_with_brace();
            if space_before_dollar || space_before_brace {
                result.push(' ');
            }
        }
    }

    if !use_multiple_lines && result.len() >= shape.width {
        None
    } else {
        Some(result)
    }
}

// This is a bit sketchy. The token rules probably need tweaking, but it works
// for some common cases. I hope the basic logic is sufficient. Note that the
// meaning of some tokens is a bit different here from usual Rust, e.g., `*`
// and `(`/`)` have special meaning.
//
// We always try and format on one line.
// FIXME: Use multi-line when every thing does not fit on one line.
fn format_macro_args(
    context: &RewriteContext,
    toks: ThinTokenStream,
    shape: Shape,
) -> Option<String> {
    if !context.config.format_macro_matchers() {
        let token_stream: TokenStream = toks.into();
        let span = span_for_token_stream(token_stream);
        return Some(match span {
            Some(span) => context.snippet(span).to_owned(),
            None => String::new(),
        });
    }
    let parsed_args = MacroArgParser::new().parse(toks)?;
    wrap_macro_args(context, &parsed_args, shape)
}

fn span_for_token_stream(token_stream: TokenStream) -> Option<Span> {
    token_stream.trees().next().map(|tt| tt.span())
}

// We should insert a space if the next token is a:
#[derive(Copy, Clone, PartialEq)]
enum SpaceState {
    Never,
    Punctuation,
    Ident, // Or ident/literal-like thing.
    Always,
}

fn force_space_before(tok: &Token) -> bool {
    debug!("tok: force_space_before {:?}", tok);

    match *tok {
        Token::Eq
        | Token::Lt
        | Token::Le
        | Token::EqEq
        | Token::Ne
        | Token::Ge
        | Token::Gt
        | Token::AndAnd
        | Token::OrOr
        | Token::Not
        | Token::Tilde
        | Token::BinOpEq(_)
        | Token::At
        | Token::RArrow
        | Token::LArrow
        | Token::FatArrow
        | Token::BinOp(_)
        | Token::Pound
        | Token::Dollar => true,
        _ => false,
    }
}

fn ident_like(tok: &Token) -> bool {
    match *tok {
        Token::Ident(..) | Token::Literal(..) | Token::Lifetime(_) => true,
        _ => false,
    }
}

fn next_space(tok: &Token) -> SpaceState {
    debug!("next_space: {:?}", tok);

    match *tok {
        Token::Not
        | Token::BinOp(BinOpToken::And)
        | Token::Tilde
        | Token::At
        | Token::Comma
        | Token::Dot
        | Token::DotDot
        | Token::DotDotDot
        | Token::DotDotEq
        | Token::DotEq
        | Token::Question => SpaceState::Punctuation,

        Token::ModSep
        | Token::Pound
        | Token::Dollar
        | Token::OpenDelim(_)
        | Token::CloseDelim(_)
        | Token::Whitespace => SpaceState::Never,

        Token::Literal(..) | Token::Ident(..) | Token::Lifetime(_) => SpaceState::Ident,

        _ => SpaceState::Always,
    }
}

/// Tries to convert a macro use into a short hand try expression. Returns None
/// when the macro is not an instance of try! (or parsing the inner expression
/// failed).
pub fn convert_try_mac(mac: &ast::Mac, context: &RewriteContext) -> Option<ast::Expr> {
    if &format!("{}", mac.node.path) == "try" {
        let ts: TokenStream = mac.node.tts.clone().into();
        let mut parser = new_parser_from_tts(context.parse_session, ts.trees().collect());

        Some(ast::Expr {
            id: ast::NodeId::new(0), // dummy value
            node: ast::ExprKind::Try(parser.parse_expr().ok()?),
            span: mac.span, // incorrect span, but shouldn't matter too much
            attrs: ThinVec::new(),
        })
    } else {
        None
    }
}

fn macro_style(mac: &ast::Mac, context: &RewriteContext) -> DelimToken {
    let snippet = context.snippet(mac.span);
    let paren_pos = snippet.find_uncommented("(").unwrap_or(usize::max_value());
    let bracket_pos = snippet.find_uncommented("[").unwrap_or(usize::max_value());
    let brace_pos = snippet.find_uncommented("{").unwrap_or(usize::max_value());

    if paren_pos < bracket_pos && paren_pos < brace_pos {
        DelimToken::Paren
    } else if bracket_pos < brace_pos {
        DelimToken::Bracket
    } else {
        DelimToken::Brace
    }
}

/// Indent each line according to the specified `indent`.
/// e.g.
///
/// ```rust,ignore
/// foo!{
/// x,
/// y,
/// foo(
///     a,
///     b,
///     c,
/// ),
/// }
/// ```
///
/// will become
///
/// ```rust,ignore
/// foo!{
///     x,
///     y,
///     foo(
///         a,
///         b,
///         c,
///     ),
/// }
/// ```
fn indent_macro_snippet(
    context: &RewriteContext,
    macro_str: &str,
    indent: Indent,
) -> Option<String> {
    let mut lines = LineClasses::new(macro_str);
    let first_line = lines.next().map(|(_, s)| s.trim_right().to_owned())?;
    let mut trimmed_lines = Vec::with_capacity(16);

    let mut veto_trim = false;
    let min_prefix_space_width = lines
        .filter_map(|(kind, line)| {
            let mut trimmed = true;
            let prefix_space_width = if is_empty_line(&line) {
                None
            } else {
                Some(get_prefix_space_width(context, &line))
            };
            let line = if veto_trim || (kind.is_string() && !line.ends_with('\\')) {
                veto_trim = kind.is_string() && !line.ends_with('\\');
                trimmed = false;
                line
            } else {
                line.trim().to_owned()
            };
            trimmed_lines.push((trimmed, line, prefix_space_width));
            prefix_space_width
        }).min()?;

    Some(
        first_line + "\n" + &trimmed_lines
            .iter()
            .map(
                |&(trimmed, ref line, prefix_space_width)| match prefix_space_width {
                    _ if !trimmed => line.to_owned(),
                    Some(original_indent_width) => {
                        let new_indent_width = indent.width() + original_indent_width
                            .saturating_sub(min_prefix_space_width);
                        let new_indent = Indent::from_width(context.config, new_indent_width);
                        format!("{}{}", new_indent.to_string(context.config), line.trim())
                    }
                    None => String::new(),
                },
            ).collect::<Vec<_>>()
            .join("\n"),
    )
}

fn get_prefix_space_width(context: &RewriteContext, s: &str) -> usize {
    let mut width = 0;
    for c in s.chars() {
        match c {
            ' ' => width += 1,
            '\t' => width += context.config.tab_spaces(),
            _ => return width,
        }
    }
    width
}

fn is_empty_line(s: &str) -> bool {
    s.is_empty() || s.chars().all(char::is_whitespace)
}

// A very simple parser that just parses a macros 2.0 definition into its branches.
// Currently we do not attempt to parse any further than that.
#[derive(new)]
struct MacroParser {
    toks: Cursor,
}

impl MacroParser {
    // (`(` ... `)` `=>` `{` ... `}`)*
    fn parse(&mut self) -> Option<Macro> {
        let mut branches = vec![];
        while self.toks.look_ahead(1).is_some() {
            branches.push(self.parse_branch()?);
        }

        Some(Macro { branches })
    }

    // `(` ... `)` `=>` `{` ... `}`
    fn parse_branch(&mut self) -> Option<MacroBranch> {
        let tok = self.toks.next()?;
        let (lo, args_paren_kind) = match tok {
            TokenTree::Token(..) => return None,
            TokenTree::Delimited(sp, ref d) => (sp.lo(), d.delim),
        };
        let args = tok.joint().into();
        match self.toks.next()? {
            TokenTree::Token(_, Token::FatArrow) => {}
            _ => return None,
        }
        let (mut hi, body, whole_body) = match self.toks.next()? {
            TokenTree::Token(..) => return None,
            TokenTree::Delimited(sp, _) => {
                let data = sp.data();
                (
                    data.hi,
                    Span::new(data.lo + BytePos(1), data.hi - BytePos(1), data.ctxt),
                    sp,
                )
            }
        };
        if let Some(TokenTree::Token(sp, Token::Semi)) = self.toks.look_ahead(0) {
            self.toks.next();
            hi = sp.hi();
        }
        Some(MacroBranch {
            span: mk_sp(lo, hi),
            args_paren_kind,
            args,
            body,
            whole_body,
        })
    }
}

// A parsed macros 2.0 macro definition.
struct Macro {
    branches: Vec<MacroBranch>,
}

// FIXME: it would be more efficient to use references to the token streams
// rather than clone them, if we can make the borrowing work out.
struct MacroBranch {
    span: Span,
    args_paren_kind: DelimToken,
    args: ThinTokenStream,
    body: Span,
    whole_body: Span,
}

impl MacroBranch {
    fn rewrite(
        &self,
        context: &RewriteContext,
        shape: Shape,
        multi_branch_style: bool,
    ) -> Option<String> {
        // Only attempt to format function-like macros.
        if self.args_paren_kind != DelimToken::Paren {
            // FIXME(#1539): implement for non-sugared macros.
            return None;
        }

        // 5 = " => {"
        let mut result = format_macro_args(context, self.args.clone(), shape.sub_width(5)?)?;

        if multi_branch_style {
            result += " =>";
        }

        if !context.config.format_macro_bodies() {
            result += " ";
            result += context.snippet(self.whole_body);
            return Some(result);
        }

        // The macro body is the most interesting part. It might end up as various
        // AST nodes, but also has special variables (e.g, `$foo`) which can't be
        // parsed as regular Rust code (and note that these can be escaped using
        // `$$`). We'll try and format like an AST node, but we'll substitute
        // variables for new names with the same length first.

        let old_body = context.snippet(self.body).trim();
        let (body_str, substs) = replace_names(old_body)?;
        let has_block_body = old_body.starts_with('{');

        let mut config = context.config.clone();
        config.set().hide_parse_errors(true);

        result += " {";

        let body_indent = if has_block_body {
            shape.indent
        } else {
            shape.indent.block_indent(&config)
        };
        let new_width = config.max_width() - body_indent.width();
        config.set().max_width(new_width);

        // First try to format as items, then as statements.
        let new_body = match ::format_snippet(&body_str, &config) {
            Some(new_body) => new_body,
            None => {
                let new_width = new_width + config.tab_spaces();
                config.set().max_width(new_width);
                match ::format_code_block(&body_str, &config) {
                    Some(new_body) => new_body,
                    None => return None,
                }
            }
        };
        let new_body = wrap_str(new_body, config.max_width(), shape)?;

        // Indent the body since it is in a block.
        let indent_str = body_indent.to_string(&config);
        let mut new_body = LineClasses::new(new_body.trim_right())
            .fold(
                (String::new(), true),
                |(mut s, need_indent), (kind, ref l)| {
                    if !l.is_empty() && need_indent {
                        s += &indent_str;
                    }
                    (s + l + "\n", !kind.is_string() || l.ends_with('\\'))
                },
            ).0;

        // Undo our replacement of macro variables.
        // FIXME: this could be *much* more efficient.
        for (old, new) in &substs {
            if old_body.find(new).is_some() {
                debug!("rewrite_macro_def: bailing matching variable: `{}`", new);
                return None;
            }
            new_body = new_body.replace(new, old);
        }

        if has_block_body {
            result += new_body.trim();
        } else if !new_body.is_empty() {
            result += "\n";
            result += &new_body;
            result += &shape.indent.to_string(&config);
        }

        result += "}";

        Some(result)
    }
}

/// Format `lazy_static!` from https://crates.io/crates/lazy_static.
///
/// # Expected syntax
///
/// ```ignore
/// lazy_static! {
///     [pub] static ref NAME_1: TYPE_1 = EXPR_1;
///     [pub] static ref NAME_2: TYPE_2 = EXPR_2;
///     ...
///     [pub] static ref NAME_N: TYPE_N = EXPR_N;
/// }
/// ```
fn format_lazy_static(context: &RewriteContext, shape: Shape, ts: &TokenStream) -> Option<String> {
    let mut result = String::with_capacity(1024);
    let mut parser = new_parser_from_tts(context.parse_session, ts.trees().collect());
    let nested_shape = shape
        .block_indent(context.config.tab_spaces())
        .with_max_width(context.config);

    result.push_str("lazy_static! {");
    result.push_str(&nested_shape.indent.to_string_with_newline(context.config));

    macro parse_or($method:ident $(,)* $($arg:expr),* $(,)*) {
        match parser.$method($($arg,)*) {
            Ok(val) => {
                if parser.sess.span_diagnostic.has_errors() {
                    parser.sess.span_diagnostic.reset_err_count();
                    return None;
                } else {
                    val
                }
            }
            Err(mut err) => {
                err.cancel();
                parser.sess.span_diagnostic.reset_err_count();
                return None;
            }
        }
    }

    while parser.token != Token::Eof {
        // Parse a `lazy_static!` item.
        let vis = ::utils::format_visibility(context, &parse_or!(parse_visibility, false));
        parser.eat_keyword(symbol::keywords::Static);
        parser.eat_keyword(symbol::keywords::Ref);
        let id = parse_or!(parse_ident);
        parser.eat(&Token::Colon);
        let ty = parse_or!(parse_ty);
        parser.eat(&Token::Eq);
        let expr = parse_or!(parse_expr);
        parser.eat(&Token::Semi);

        // Rewrite as a static item.
        let mut stmt = String::with_capacity(128);
        stmt.push_str(&format!(
            "{}static ref {}: {} =",
            vis,
            id,
            ty.rewrite(context, nested_shape)?
        ));
        result.push_str(&::expr::rewrite_assign_rhs(
            context,
            stmt,
            &*expr,
            nested_shape.sub_width(1)?,
        )?);
        result.push(';');
        if parser.token != Token::Eof {
            result.push_str(&nested_shape.indent.to_string_with_newline(context.config));
        }
    }

    result.push_str(&shape.indent.to_string_with_newline(context.config));
    result.push('}');

    Some(result)
}
