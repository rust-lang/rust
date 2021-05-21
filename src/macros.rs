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
use std::panic::{catch_unwind, AssertUnwindSafe};

use rustc_ast::token::{BinOpToken, DelimToken, Token, TokenKind};
use rustc_ast::tokenstream::{Cursor, Spacing, TokenStream, TokenTree};
use rustc_ast::{ast, ptr};
use rustc_ast_pretty::pprust;
use rustc_parse::parser::{ForceCollect, Parser};
use rustc_parse::{stream_to_parser, MACRO_ARGUMENTS};
use rustc_span::{
    symbol::{self, kw},
    BytePos, Span, Symbol, DUMMY_SP,
};

use crate::comment::{
    contains_comment, CharClasses, FindUncommented, FullCodeCharKind, LineClasses,
};
use crate::config::lists::*;
use crate::expr::rewrite_array;
use crate::lists::{itemize_list, write_list, ListFormatting};
use crate::overflow;
use crate::rewrite::{Rewrite, RewriteContext};
use crate::shape::{Indent, Shape};
use crate::source_map::SpanUtils;
use crate::spanned::Spanned;
use crate::utils::{
    format_visibility, indent_next_line, is_empty_line, mk_sp, remove_trailing_white_spaces,
    rewrite_ident, trim_left_preserve_layout, wrap_str, NodeIdExt,
};
use crate::visitor::FmtVisitor;

const FORCED_BRACKET_MACROS: &[&str] = &["vec!"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MacroPosition {
    Item,
    Statement,
    Expression,
    Pat,
}

#[derive(Debug)]
pub(crate) enum MacroArg {
    Expr(ptr::P<ast::Expr>),
    Ty(ptr::P<ast::Ty>),
    Pat(ptr::P<ast::Pat>),
    Item(ptr::P<ast::Item>),
    Keyword(symbol::Ident, Span),
}

impl MacroArg {
    fn is_item(&self) -> bool {
        match self {
            MacroArg::Item(..) => true,
            _ => false,
        }
    }
}

impl Rewrite for ast::Item {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        let mut visitor = crate::visitor::FmtVisitor::from_context(context);
        visitor.block_indent = shape.indent;
        visitor.last_pos = self.span().lo();
        visitor.visit_item(self);
        Some(visitor.buffer.to_owned())
    }
}

impl Rewrite for MacroArg {
    fn rewrite(&self, context: &RewriteContext<'_>, shape: Shape) -> Option<String> {
        match *self {
            MacroArg::Expr(ref expr) => expr.rewrite(context, shape),
            MacroArg::Ty(ref ty) => ty.rewrite(context, shape),
            MacroArg::Pat(ref pat) => pat.rewrite(context, shape),
            MacroArg::Item(ref item) => item.rewrite(context, shape),
            MacroArg::Keyword(ident, _) => Some(ident.name.to_string()),
        }
    }
}

fn build_parser<'a>(context: &RewriteContext<'a>, cursor: Cursor) -> Parser<'a> {
    stream_to_parser(
        context.parse_sess.inner(),
        cursor.collect(),
        MACRO_ARGUMENTS,
    )
}

fn parse_macro_arg<'a, 'b: 'a>(parser: &'a mut Parser<'b>) -> Option<MacroArg> {
    macro_rules! parse_macro_arg {
        ($macro_arg:ident, $parser:expr, $f:expr) => {
            let mut cloned_parser = (*parser).clone();
            match $parser(&mut cloned_parser) {
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

    parse_macro_arg!(
        Expr,
        |parser: &mut rustc_parse::parser::Parser<'b>| parser.parse_expr(),
        |x: ptr::P<ast::Expr>| Some(x)
    );
    parse_macro_arg!(
        Ty,
        |parser: &mut rustc_parse::parser::Parser<'b>| parser.parse_ty(),
        |x: ptr::P<ast::Ty>| Some(x)
    );
    parse_macro_arg!(
        Pat,
        |parser: &mut rustc_parse::parser::Parser<'b>| parser.parse_pat_no_top_alt(None),
        |x: ptr::P<ast::Pat>| Some(x)
    );
    // `parse_item` returns `Option<ptr::P<ast::Item>>`.
    parse_macro_arg!(
        Item,
        |parser: &mut rustc_parse::parser::Parser<'b>| parser.parse_item(ForceCollect::No),
        |x: Option<ptr::P<ast::Item>>| x
    );

    None
}

/// Rewrite macro name without using pretty-printer if possible.
fn rewrite_macro_name(
    context: &RewriteContext<'_>,
    path: &ast::Path,
    extra_ident: Option<symbol::Ident>,
) -> String {
    let name = if path.segments.len() == 1 {
        // Avoid using pretty-printer in the common case.
        format!("{}!", rewrite_ident(context, path.segments[0].ident))
    } else {
        format!("{}!", pprust::path_to_string(path))
    };
    match extra_ident {
        Some(ident) if ident.name != kw::Empty => format!("{} {}", name, ident),
        _ => name,
    }
}

// Use this on failing to format the macro call.
fn return_macro_parse_failure_fallback(
    context: &RewriteContext<'_>,
    indent: Indent,
    span: Span,
) -> Option<String> {
    // Mark this as a failure however we format it
    context.macro_rewrite_failure.replace(true);

    // Heuristically determine whether the last line of the macro uses "Block" style
    // rather than using "Visual" style, or another indentation style.
    let is_like_block_indent_style = context
        .snippet(span)
        .lines()
        .last()
        .map(|closing_line| {
            closing_line
                .trim()
                .chars()
                .all(|ch| matches!(ch, '}' | ')' | ']'))
        })
        .unwrap_or(false);
    if is_like_block_indent_style {
        return trim_left_preserve_layout(context.snippet(span), indent, &context.config);
    }

    context.skipped_range.borrow_mut().push((
        context.parse_sess.line_of_byte_pos(span.lo()),
        context.parse_sess.line_of_byte_pos(span.hi()),
    ));

    // Return the snippet unmodified if the macro is not block-like
    Some(context.snippet(span).to_owned())
}

pub(crate) fn rewrite_macro(
    mac: &ast::MacCall,
    extra_ident: Option<symbol::Ident>,
    context: &RewriteContext<'_>,
    shape: Shape,
    position: MacroPosition,
) -> Option<String> {
    let should_skip = context
        .skip_context
        .skip_macro(&context.snippet(mac.path.span).to_owned());
    if should_skip {
        None
    } else {
        let guard = context.enter_macro();
        let result = catch_unwind(AssertUnwindSafe(|| {
            rewrite_macro_inner(
                mac,
                extra_ident,
                context,
                shape,
                position,
                guard.is_nested(),
            )
        }));
        match result {
            Err(..) | Ok(None) => {
                context.macro_rewrite_failure.replace(true);
                None
            }
            Ok(rw) => rw,
        }
    }
}

fn check_keyword<'a, 'b: 'a>(parser: &'a mut Parser<'b>) -> Option<MacroArg> {
    for &keyword in RUST_KW.iter() {
        if parser.token.is_keyword(keyword)
            && parser.look_ahead(1, |t| {
                t.kind == TokenKind::Eof
                    || t.kind == TokenKind::Comma
                    || t.kind == TokenKind::CloseDelim(DelimToken::NoDelim)
            })
        {
            parser.bump();
            return Some(MacroArg::Keyword(
                symbol::Ident::with_dummy_span(keyword),
                parser.prev_token.span,
            ));
        }
    }
    None
}

fn rewrite_macro_inner(
    mac: &ast::MacCall,
    extra_ident: Option<symbol::Ident>,
    context: &RewriteContext<'_>,
    shape: Shape,
    position: MacroPosition,
    is_nested_macro: bool,
) -> Option<String> {
    if context.config.use_try_shorthand() {
        if let Some(expr) = convert_try_mac(mac, context) {
            context.leave_macro();
            return expr.rewrite(context, shape);
        }
    }

    let original_style = macro_style(mac, context);

    let macro_name = rewrite_macro_name(context, &mac.path, extra_ident);

    let style = if FORCED_BRACKET_MACROS.contains(&&macro_name[..]) && !is_nested_macro {
        DelimToken::Bracket
    } else {
        original_style
    };

    let ts = mac.args.inner_tokens();
    let has_comment = contains_comment(context.snippet(mac.span()));
    if ts.is_empty() && !has_comment {
        return match style {
            DelimToken::Paren if position == MacroPosition::Item => {
                Some(format!("{}();", macro_name))
            }
            DelimToken::Bracket if position == MacroPosition::Item => {
                Some(format!("{}[];", macro_name))
            }
            DelimToken::Paren => Some(format!("{}()", macro_name)),
            DelimToken::Bracket => Some(format!("{}[]", macro_name)),
            DelimToken::Brace => Some(format!("{} {{}}", macro_name)),
            _ => unreachable!(),
        };
    }
    // Format well-known macros which cannot be parsed as a valid AST.
    if macro_name == "lazy_static!" && !has_comment {
        if let success @ Some(..) = format_lazy_static(context, shape, &ts) {
            return success;
        }
    }

    let mut parser = build_parser(context, ts.trees());
    let mut arg_vec = Vec::new();
    let mut vec_with_semi = false;
    let mut trailing_comma = false;

    if DelimToken::Brace != style {
        loop {
            if let Some(arg) = check_keyword(&mut parser) {
                arg_vec.push(arg);
            } else if let Some(arg) = parse_macro_arg(&mut parser) {
                arg_vec.push(arg);
            } else {
                return return_macro_parse_failure_fallback(context, shape.indent, mac.span());
            }

            match parser.token.kind {
                TokenKind::Eof => break,
                TokenKind::Comma => (),
                TokenKind::Semi => {
                    // Try to parse `vec![expr; expr]`
                    if FORCED_BRACKET_MACROS.contains(&&macro_name[..]) {
                        parser.bump();
                        if parser.token.kind != TokenKind::Eof {
                            match parse_macro_arg(&mut parser) {
                                Some(arg) => {
                                    arg_vec.push(arg);
                                    parser.bump();
                                    if parser.token.kind == TokenKind::Eof && arg_vec.len() == 2 {
                                        vec_with_semi = true;
                                        break;
                                    }
                                }
                                None => {
                                    return return_macro_parse_failure_fallback(
                                        context,
                                        shape.indent,
                                        mac.span(),
                                    );
                                }
                            }
                        }
                    }
                    return return_macro_parse_failure_fallback(context, shape.indent, mac.span());
                }
                _ if arg_vec.last().map_or(false, MacroArg::is_item) => continue,
                _ => return return_macro_parse_failure_fallback(context, shape.indent, mac.span()),
            }

            parser.bump();

            if parser.token.kind == TokenKind::Eof {
                trailing_comma = true;
                break;
            }
        }
    }

    if !arg_vec.is_empty() && arg_vec.iter().all(MacroArg::is_item) {
        return rewrite_macro_with_items(
            context,
            &arg_vec,
            &macro_name,
            shape,
            style,
            position,
            mac.span(),
        );
    }

    match style {
        DelimToken::Paren => {
            // Handle special case: `vec!(expr; expr)`
            if vec_with_semi {
                handle_vec_semi(context, shape, arg_vec, macro_name, style)
            } else {
                // Format macro invocation as function call, preserve the trailing
                // comma because not all macros support them.
                overflow::rewrite_with_parens(
                    context,
                    &macro_name,
                    arg_vec.iter(),
                    shape,
                    mac.span(),
                    context.config.fn_call_width(),
                    if trailing_comma {
                        Some(SeparatorTactic::Always)
                    } else {
                        Some(SeparatorTactic::Never)
                    },
                )
                .map(|rw| match position {
                    MacroPosition::Item => format!("{};", rw),
                    _ => rw,
                })
            }
        }
        DelimToken::Bracket => {
            // Handle special case: `vec![expr; expr]`
            if vec_with_semi {
                handle_vec_semi(context, shape, arg_vec, macro_name, style)
            } else {
                // If we are rewriting `vec!` macro or other special macros,
                // then we can rewrite this as a usual array literal.
                // Otherwise, we must preserve the original existence of trailing comma.
                let macro_name = &macro_name.as_str();
                let mut force_trailing_comma = if trailing_comma {
                    Some(SeparatorTactic::Always)
                } else {
                    Some(SeparatorTactic::Never)
                };
                if FORCED_BRACKET_MACROS.contains(macro_name) && !is_nested_macro {
                    context.leave_macro();
                    if context.use_block_indent() {
                        force_trailing_comma = Some(SeparatorTactic::Vertical);
                    };
                }
                let rewrite = rewrite_array(
                    macro_name,
                    arg_vec.iter(),
                    mac.span(),
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
            // For macro invocations with braces, always put a space between
            // the `macro_name!` and `{ /* macro_body */ }` but skip modifying
            // anything in between the braces (for now).
            let snippet = context.snippet(mac.span()).trim_start_matches(|c| c != '{');
            match trim_left_preserve_layout(snippet, shape.indent, &context.config) {
                Some(macro_body) => Some(format!("{} {}", macro_name, macro_body)),
                None => Some(format!("{} {}", macro_name, snippet)),
            }
        }
        _ => unreachable!(),
    }
}

fn handle_vec_semi(
    context: &RewriteContext<'_>,
    shape: Shape,
    arg_vec: Vec<MacroArg>,
    macro_name: String,
    delim_token: DelimToken,
) -> Option<String> {
    let (left, right) = match delim_token {
        DelimToken::Paren => ("(", ")"),
        DelimToken::Bracket => ("[", "]"),
        _ => unreachable!(),
    };

    let mac_shape = shape.offset_left(macro_name.len())?;
    // 8 = `vec![]` + `; ` or `vec!()` + `; `
    let total_overhead = 8;
    let nested_shape = mac_shape.block_indent(context.config.tab_spaces());
    let lhs = arg_vec[0].rewrite(context, nested_shape)?;
    let rhs = arg_vec[1].rewrite(context, nested_shape)?;
    if !lhs.contains('\n')
        && !rhs.contains('\n')
        && lhs.len() + rhs.len() + total_overhead <= shape.width
    {
        // macro_name(lhs; rhs) or macro_name[lhs; rhs]
        Some(format!("{}{}{}; {}{}", macro_name, left, lhs, rhs, right))
    } else {
        // macro_name(\nlhs;\nrhs\n) or macro_name[\nlhs;\nrhs\n]
        Some(format!(
            "{}{}{}{};{}{}{}{}",
            macro_name,
            left,
            nested_shape.indent.to_string_with_newline(context.config),
            lhs,
            nested_shape.indent.to_string_with_newline(context.config),
            rhs,
            shape.indent.to_string_with_newline(context.config),
            right
        ))
    }
}

pub(crate) fn rewrite_macro_def(
    context: &RewriteContext<'_>,
    shape: Shape,
    indent: Indent,
    def: &ast::MacroDef,
    ident: symbol::Ident,
    vis: &ast::Visibility,
    span: Span,
) -> Option<String> {
    let snippet = Some(remove_trailing_white_spaces(context.snippet(span)));
    if snippet.as_ref().map_or(true, |s| s.ends_with(';')) {
        return snippet;
    }

    let ts = def.body.inner_tokens();
    let mut parser = MacroParser::new(ts.into_trees());
    let parsed_def = match parser.parse() {
        Some(def) => def,
        None => return snippet,
    };

    let mut result = if def.macro_rules {
        String::from("macro_rules!")
    } else {
        format!("{}macro", format_visibility(context, vis))
    };

    result += " ";
    result += rewrite_ident(context, ident);

    let multi_branch_style = def.macro_rules || parsed_def.branches.len() != 1;

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
        |branch| match branch.rewrite(context, arm_shape, multi_branch_style) {
            Some(v) => Some(v),
            // if the rewrite returned None because a macro could not be rewritten, then return the
            // original body
            None if context.macro_rewrite_failure.get() => {
                Some(context.snippet(branch.body).trim().to_string())
            }
            None => None,
        },
        context.snippet_provider.span_after(span, "{"),
        span.hi(),
        false,
    )
    .collect::<Vec<_>>();

    let fmt = ListFormatting::new(arm_shape, context.config)
        .separator(if def.macro_rules { ";" } else { "" })
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
    let mut new_name = "$".repeat(dollar_count - 1);
    let mut old_name = "$".repeat(dollar_count);

    new_name.push('z');
    new_name.push_str(name);
    old_name.push_str(name);

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
    /// e.g., `$x: expr`.
    MetaVariable(Symbol, String),
    /// e.g., `$($foo: expr),*`
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
    /// e.g., `[derive(Debug)]`
    Delimited(DelimToken, Vec<ParsedMacroArg>),
    /// A possible separator. e.g., `,` or `;`.
    Separator(String, String),
    /// Other random stuff that does not fit to other kinds.
    /// e.g., `== foo` in `($x: expr == foo)`.
    Other(String, String),
}

fn delim_token_to_str(
    context: &RewriteContext<'_>,
    delim_token: DelimToken,
    shape: Shape,
    use_multiple_lines: bool,
    inner_is_empty: bool,
) -> (String, String) {
    let (lhs, rhs) = match delim_token {
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
        matches!(
            *self,
            MacroArgKind::Repeat(DelimToken::Brace, _, _, _)
                | MacroArgKind::Delimited(DelimToken::Brace, _)
        )
    }

    fn starts_with_dollar(&self) -> bool {
        matches!(
            *self,
            MacroArgKind::Repeat(..) | MacroArgKind::MetaVariable(..)
        )
    }

    fn ends_with_space(&self) -> bool {
        matches!(*self, MacroArgKind::Separator(..))
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
        context: &RewriteContext<'_>,
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
            MacroArgKind::MetaVariable(ty, ref name) => Some(format!("${}:{}", name, ty)),
            MacroArgKind::Repeat(delim_tok, ref args, ref another, ref tok) => {
                let (lhs, inner, rhs) = rewrite_delimited_inner(delim_tok, args)?;
                let another = another
                    .as_ref()
                    .and_then(|a| a.rewrite(context, shape, use_multiple_lines))
                    .unwrap_or_else(|| "".to_owned());
                let repeat_tok = pprust::token_to_string(tok);

                Some(format!("${}{}{}{}{}", lhs, inner, rhs, another, repeat_tok))
            }
            MacroArgKind::Delimited(delim_tok, ref args) => {
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
}

impl ParsedMacroArg {
    fn rewrite(
        &self,
        context: &RewriteContext<'_>,
        shape: Shape,
        use_multiple_lines: bool,
    ) -> Option<String> {
        self.kind.rewrite(context, shape, use_multiple_lines)
    }
}

/// Parses macro arguments on macro def.
struct MacroArgParser {
    /// Either a name of the next metavariable, a separator, or junk.
    buf: String,
    /// The first token of the current buffer.
    start_tok: Token,
    /// `true` if we are parsing a metavariable or a repeat.
    is_meta_var: bool,
    /// The last token parsed.
    last_tok: Token,
    /// Holds the parsed arguments.
    result: Vec<ParsedMacroArg>,
}

fn last_tok(tt: &TokenTree) -> Token {
    match *tt {
        TokenTree::Token(ref t) => t.clone(),
        TokenTree::Delimited(delim_span, delim, _) => Token {
            kind: TokenKind::CloseDelim(delim),
            span: delim_span.close,
        },
    }
}

impl MacroArgParser {
    fn new() -> MacroArgParser {
        MacroArgParser {
            buf: String::new(),
            is_meta_var: false,
            last_tok: Token {
                kind: TokenKind::Eof,
                span: DUMMY_SP,
            },
            start_tok: Token {
                kind: TokenKind::Eof,
                span: DUMMY_SP,
            },
            result: vec![],
        }
    }

    fn set_last_tok(&mut self, tok: &TokenTree) {
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
        });
        self.buf.clear();
    }

    fn add_meta_variable(&mut self, iter: &mut Cursor) -> Option<()> {
        match iter.next() {
            Some(TokenTree::Token(Token {
                kind: TokenKind::Ident(name, _),
                ..
            })) => {
                self.result.push(ParsedMacroArg {
                    kind: MacroArgKind::MetaVariable(name, self.buf.clone()),
                });

                self.buf.clear();
                self.is_meta_var = false;
                Some(())
            }
            _ => None,
        }
    }

    fn add_delimited(&mut self, inner: Vec<ParsedMacroArg>, delim: DelimToken) {
        self.result.push(ParsedMacroArg {
            kind: MacroArgKind::Delimited(delim, inner),
        });
    }

    // $($foo: expr),?
    fn add_repeat(
        &mut self,
        inner: Vec<ParsedMacroArg>,
        delim: DelimToken,
        iter: &mut Cursor,
    ) -> Option<()> {
        let mut buffer = String::new();
        let mut first = true;

        // Parse '*', '+' or '?.
        for tok in iter {
            self.set_last_tok(&tok);
            if first {
                first = false;
            }

            match tok {
                TokenTree::Token(Token {
                    kind: TokenKind::BinOp(BinOpToken::Plus),
                    ..
                })
                | TokenTree::Token(Token {
                    kind: TokenKind::Question,
                    ..
                })
                | TokenTree::Token(Token {
                    kind: TokenKind::BinOp(BinOpToken::Star),
                    ..
                }) => {
                    break;
                }
                TokenTree::Token(ref t) => {
                    buffer.push_str(&pprust::token_to_string(&t));
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
            }))
        };

        self.result.push(ParsedMacroArg {
            kind: MacroArgKind::Repeat(delim, inner, another, self.last_tok.clone()),
        });
        Some(())
    }

    fn update_buffer(&mut self, t: &Token) {
        if self.buf.is_empty() {
            self.start_tok = t.clone();
        } else {
            let needs_space = match next_space(&self.last_tok.kind) {
                SpaceState::Ident => ident_like(t),
                SpaceState::Punctuation => !ident_like(t),
                SpaceState::Always => true,
                SpaceState::Never => false,
            };
            if force_space_before(&t.kind) || needs_space {
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
            if self.start_tok.kind == TokenKind::Colon {
                return true;
            }
        }

        if force_space_before(&self.start_tok.kind) {
            return true;
        }

        false
    }

    /// Returns a collection of parsed macro def's arguments.
    fn parse(mut self, tokens: TokenStream) -> Option<Vec<ParsedMacroArg>> {
        let mut iter = tokens.trees();

        while let Some(tok) = iter.next() {
            match tok {
                TokenTree::Token(Token {
                    kind: TokenKind::Dollar,
                    span,
                }) => {
                    // We always want to add a separator before meta variables.
                    if !self.buf.is_empty() {
                        self.add_separator();
                    }

                    // Start keeping the name of this metavariable in the buffer.
                    self.is_meta_var = true;
                    self.start_tok = Token {
                        kind: TokenKind::Dollar,
                        span,
                    };
                }
                TokenTree::Token(Token {
                    kind: TokenKind::Colon,
                    ..
                }) if self.is_meta_var => {
                    self.add_meta_variable(&mut iter)?;
                }
                TokenTree::Token(ref t) => self.update_buffer(t),
                TokenTree::Delimited(_delimited_span, delimited, ref tts) => {
                    if !self.buf.is_empty() {
                        if next_space(&self.last_tok.kind) == SpaceState::Always {
                            self.add_separator();
                        } else {
                            self.add_other();
                        }
                    }

                    // Parse the stuff inside delimiters.
                    let parser = MacroArgParser::new();
                    let delimited_arg = parser.parse(tts.clone())?;

                    if self.is_meta_var {
                        self.add_repeat(delimited_arg, delimited, &mut iter)?;
                        self.is_meta_var = false;
                    } else {
                        self.add_delimited(delimited_arg, delimited);
                    }
                }
            }

            self.set_last_tok(&tok);
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
    context: &RewriteContext<'_>,
    args: &[ParsedMacroArg],
    shape: Shape,
) -> Option<String> {
    wrap_macro_args_inner(context, args, shape, false)
        .or_else(|| wrap_macro_args_inner(context, args, shape, true))
}

fn wrap_macro_args_inner(
    context: &RewriteContext<'_>,
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
    context: &RewriteContext<'_>,
    token_stream: TokenStream,
    shape: Shape,
) -> Option<String> {
    if !context.config.format_macro_matchers() {
        let span = span_for_token_stream(&token_stream);
        return Some(match span {
            Some(span) => context.snippet(span).to_owned(),
            None => String::new(),
        });
    }
    let parsed_args = MacroArgParser::new().parse(token_stream)?;
    wrap_macro_args(context, &parsed_args, shape)
}

fn span_for_token_stream(token_stream: &TokenStream) -> Option<Span> {
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

fn force_space_before(tok: &TokenKind) -> bool {
    debug!("tok: force_space_before {:?}", tok);

    match tok {
        TokenKind::Eq
        | TokenKind::Lt
        | TokenKind::Le
        | TokenKind::EqEq
        | TokenKind::Ne
        | TokenKind::Ge
        | TokenKind::Gt
        | TokenKind::AndAnd
        | TokenKind::OrOr
        | TokenKind::Not
        | TokenKind::Tilde
        | TokenKind::BinOpEq(_)
        | TokenKind::At
        | TokenKind::RArrow
        | TokenKind::LArrow
        | TokenKind::FatArrow
        | TokenKind::BinOp(_)
        | TokenKind::Pound
        | TokenKind::Dollar => true,
        _ => false,
    }
}

fn ident_like(tok: &Token) -> bool {
    matches!(
        tok.kind,
        TokenKind::Ident(..) | TokenKind::Literal(..) | TokenKind::Lifetime(_)
    )
}

fn next_space(tok: &TokenKind) -> SpaceState {
    debug!("next_space: {:?}", tok);

    match tok {
        TokenKind::Not
        | TokenKind::BinOp(BinOpToken::And)
        | TokenKind::Tilde
        | TokenKind::At
        | TokenKind::Comma
        | TokenKind::Dot
        | TokenKind::DotDot
        | TokenKind::DotDotDot
        | TokenKind::DotDotEq
        | TokenKind::Question => SpaceState::Punctuation,

        TokenKind::ModSep
        | TokenKind::Pound
        | TokenKind::Dollar
        | TokenKind::OpenDelim(_)
        | TokenKind::CloseDelim(_) => SpaceState::Never,

        TokenKind::Literal(..) | TokenKind::Ident(..) | TokenKind::Lifetime(_) => SpaceState::Ident,

        _ => SpaceState::Always,
    }
}

/// Tries to convert a macro use into a short hand try expression. Returns `None`
/// when the macro is not an instance of `try!` (or parsing the inner expression
/// failed).
pub(crate) fn convert_try_mac(
    mac: &ast::MacCall,
    context: &RewriteContext<'_>,
) -> Option<ast::Expr> {
    let path = &pprust::path_to_string(&mac.path);
    if path == "try" || path == "r#try" {
        let ts = mac.args.inner_tokens();
        let mut parser = build_parser(context, ts.trees());

        Some(ast::Expr {
            id: ast::NodeId::root(), // dummy value
            kind: ast::ExprKind::Try(parser.parse_expr().ok()?),
            span: mac.span(), // incorrect span, but shouldn't matter too much
            attrs: ast::AttrVec::new(),
            tokens: None,
        })
    } else {
        None
    }
}

pub(crate) fn macro_style(mac: &ast::MacCall, context: &RewriteContext<'_>) -> DelimToken {
    let snippet = context.snippet(mac.span());
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
            TokenTree::Delimited(delimited_span, d, _) => (delimited_span.open.lo(), d),
        };
        let args = TokenStream::new(vec![(tok, Spacing::Joint)]);
        match self.toks.next()? {
            TokenTree::Token(Token {
                kind: TokenKind::FatArrow,
                ..
            }) => {}
            _ => return None,
        }
        let (mut hi, body, whole_body) = match self.toks.next()? {
            TokenTree::Token(..) => return None,
            TokenTree::Delimited(delimited_span, ..) => {
                let data = delimited_span.entire().data();
                (
                    data.hi,
                    Span::new(data.lo + BytePos(1), data.hi - BytePos(1), data.ctxt),
                    delimited_span.entire(),
                )
            }
        };
        if let Some(TokenTree::Token(Token {
            kind: TokenKind::Semi,
            span,
        })) = self.toks.look_ahead(0)
        {
            hi = span.hi();
            self.toks.next();
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
    args: TokenStream,
    body: Span,
    whole_body: Span,
}

impl MacroBranch {
    fn rewrite(
        &self,
        context: &RewriteContext<'_>,
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
        let new_body_snippet = match crate::format_snippet(&body_str, &config, true) {
            Some(new_body) => new_body,
            None => {
                let new_width = new_width + config.tab_spaces();
                config.set().max_width(new_width);
                match crate::format_code_block(&body_str, &config, true) {
                    Some(new_body) => new_body,
                    None => return None,
                }
            }
        };
        let new_body = wrap_str(
            new_body_snippet.snippet.to_string(),
            config.max_width(),
            shape,
        )?;

        // Indent the body since it is in a block.
        let indent_str = body_indent.to_string(&config);
        let mut new_body = LineClasses::new(new_body.trim_end())
            .enumerate()
            .fold(
                (String::new(), true),
                |(mut s, need_indent), (i, (kind, ref l))| {
                    if !is_empty_line(l)
                        && need_indent
                        && !new_body_snippet.is_line_non_formatted(i + 1)
                    {
                        s += &indent_str;
                    }
                    (s + l + "\n", indent_next_line(kind, &l, &config))
                },
            )
            .0;

        // Undo our replacement of macro variables.
        // FIXME: this could be *much* more efficient.
        for (old, new) in &substs {
            if old_body.contains(new) {
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

/// Format `lazy_static!` from <https://crates.io/crates/lazy_static>.
///
/// # Expected syntax
///
/// ```text
/// lazy_static! {
///     [pub] static ref NAME_1: TYPE_1 = EXPR_1;
///     [pub] static ref NAME_2: TYPE_2 = EXPR_2;
///     ...
///     [pub] static ref NAME_N: TYPE_N = EXPR_N;
/// }
/// ```
fn format_lazy_static(
    context: &RewriteContext<'_>,
    shape: Shape,
    ts: &TokenStream,
) -> Option<String> {
    let mut result = String::with_capacity(1024);
    let mut parser = build_parser(context, ts.trees());
    let nested_shape = shape
        .block_indent(context.config.tab_spaces())
        .with_max_width(context.config);

    result.push_str("lazy_static! {");
    result.push_str(&nested_shape.indent.to_string_with_newline(context.config));

    macro_rules! parse_or {
        ($method:ident $(,)* $($arg:expr),* $(,)*) => {
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
    }

    while parser.token.kind != TokenKind::Eof {
        // Parse a `lazy_static!` item.
        let vis = crate::utils::format_visibility(
            context,
            &parse_or!(parse_visibility, rustc_parse::parser::FollowedByType::No),
        );
        parser.eat_keyword(kw::Static);
        parser.eat_keyword(kw::Ref);
        let id = parse_or!(parse_ident);
        parser.eat(&TokenKind::Colon);
        let ty = parse_or!(parse_ty);
        parser.eat(&TokenKind::Eq);
        let expr = parse_or!(parse_expr);
        parser.eat(&TokenKind::Semi);

        // Rewrite as a static item.
        let mut stmt = String::with_capacity(128);
        stmt.push_str(&format!(
            "{}static ref {}: {} =",
            vis,
            id,
            ty.rewrite(context, nested_shape)?
        ));
        result.push_str(&crate::expr::rewrite_assign_rhs(
            context,
            stmt,
            &*expr,
            nested_shape.sub_width(1)?,
        )?);
        result.push(';');
        if parser.token.kind != TokenKind::Eof {
            result.push_str(&nested_shape.indent.to_string_with_newline(context.config));
        }
    }

    result.push_str(&shape.indent.to_string_with_newline(context.config));
    result.push('}');

    Some(result)
}

fn rewrite_macro_with_items(
    context: &RewriteContext<'_>,
    items: &[MacroArg],
    macro_name: &str,
    shape: Shape,
    style: DelimToken,
    position: MacroPosition,
    span: Span,
) -> Option<String> {
    let (opener, closer) = match style {
        DelimToken::Paren => ("(", ")"),
        DelimToken::Bracket => ("[", "]"),
        DelimToken::Brace => (" {", "}"),
        _ => return None,
    };
    let trailing_semicolon = match style {
        DelimToken::Paren | DelimToken::Bracket if position == MacroPosition::Item => ";",
        _ => "",
    };

    let mut visitor = FmtVisitor::from_context(context);
    visitor.block_indent = shape.indent.block_indent(context.config);
    visitor.last_pos = context.snippet_provider.span_after(span, opener.trim());
    for item in items {
        let item = match item {
            MacroArg::Item(item) => item,
            _ => return None,
        };
        visitor.visit_item(&item);
    }

    let mut result = String::with_capacity(256);
    result.push_str(&macro_name);
    result.push_str(opener);
    result.push_str(&visitor.block_indent.to_string_with_newline(context.config));
    result.push_str(visitor.buffer.trim());
    result.push_str(&shape.indent.to_string_with_newline(context.config));
    result.push_str(closer);
    result.push_str(trailing_semicolon);
    Some(result)
}

const RUST_KW: [Symbol; 59] = [
    kw::PathRoot,
    kw::DollarCrate,
    kw::Underscore,
    kw::As,
    kw::Box,
    kw::Break,
    kw::Const,
    kw::Continue,
    kw::Crate,
    kw::Else,
    kw::Enum,
    kw::Extern,
    kw::False,
    kw::Fn,
    kw::For,
    kw::If,
    kw::Impl,
    kw::In,
    kw::Let,
    kw::Loop,
    kw::Match,
    kw::Mod,
    kw::Move,
    kw::Mut,
    kw::Pub,
    kw::Ref,
    kw::Return,
    kw::SelfLower,
    kw::SelfUpper,
    kw::Static,
    kw::Struct,
    kw::Super,
    kw::Trait,
    kw::True,
    kw::Type,
    kw::Unsafe,
    kw::Use,
    kw::Where,
    kw::While,
    kw::Abstract,
    kw::Become,
    kw::Do,
    kw::Final,
    kw::Macro,
    kw::Override,
    kw::Priv,
    kw::Typeof,
    kw::Unsized,
    kw::Virtual,
    kw::Yield,
    kw::Dyn,
    kw::Async,
    kw::Try,
    kw::UnderscoreLifetime,
    kw::StaticLifetime,
    kw::Auto,
    kw::Catch,
    kw::Default,
    kw::Union,
];
