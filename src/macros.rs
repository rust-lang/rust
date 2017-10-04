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
// reformated.
// List-like invocations with parentheses will be formatted as function calls,
// and those with brackets will be formatted as array literals.

use syntax::ast;
use syntax::codemap::BytePos;
use syntax::parse::new_parser_from_tts;
use syntax::parse::parser::Parser;
use syntax::parse::token::Token;
use syntax::symbol;
use syntax::tokenstream::TokenStream;
use syntax::util::ThinVec;

use codemap::SpanUtils;
use comment::{contains_comment, FindUncommented};
use expr::{rewrite_array, rewrite_call_inner};
use rewrite::{Rewrite, RewriteContext};
use shape::{Indent, Shape};
use utils::mk_sp;

const FORCED_BRACKET_MACROS: &'static [&'static str] = &["vec!"];

// FIXME: use the enum from libsyntax?
#[derive(Clone, Copy, PartialEq, Eq)]
enum MacroStyle {
    Parens,
    Brackets,
    Braces,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroPosition {
    Item,
    Statement,
    Expression,
}

impl MacroStyle {
    fn opener(&self) -> &'static str {
        match *self {
            MacroStyle::Parens => "(",
            MacroStyle::Brackets => "[",
            MacroStyle::Braces => "{",
        }
    }
}

pub enum MacroArg {
    Expr(ast::Expr),
    Ty(ast::Ty),
    Pat(ast::Pat),
}

impl Rewrite for MacroArg {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        match self {
            &MacroArg::Expr(ref expr) => expr.rewrite(context, shape),
            &MacroArg::Ty(ref ty) => ty.rewrite(context, shape),
            &MacroArg::Pat(ref pat) => pat.rewrite(context, shape),
        }
    }
}

fn parse_macro_arg(parser: &mut Parser) -> Option<MacroArg> {
    macro_rules! parse_macro_arg {
        ($target:tt, $macro_arg:ident, $parser:ident) => {
            let mut cloned_parser = (*parser).clone();
            match cloned_parser.$parser() {
                Ok($target) => {
                    if parser.sess.span_diagnostic.has_errors() {
                        parser.sess.span_diagnostic.reset_err_count();
                    } else {
                        // Parsing succeeded.
                        *parser = cloned_parser;
                        return Some(MacroArg::$macro_arg((*$target).clone()));
                    }
                }
                Err(mut e) => {
                    e.cancel();
                    parser.sess.span_diagnostic.reset_err_count();
                }
            }
        }
    }

    parse_macro_arg!(expr, Expr, parse_expr);
    parse_macro_arg!(ty, Ty, parse_ty);
    parse_macro_arg!(pat, Pat, parse_pat);

    None
}

pub fn rewrite_macro(
    mac: &ast::Mac,
    extra_ident: Option<ast::Ident>,
    context: &RewriteContext,
    shape: Shape,
    position: MacroPosition,
) -> Option<String> {
    let context = &mut context.clone();
    context.inside_macro = true;
    if context.config.use_try_shorthand() {
        if let Some(expr) = convert_try_mac(mac, context) {
            return expr.rewrite(context, shape);
        }
    }

    let original_style = macro_style(mac, context);

    let macro_name = match extra_ident {
        None => format!("{}!", mac.node.path),
        Some(ident) => if ident == symbol::keywords::Invalid.ident() {
            format!("{}!", mac.node.path)
        } else {
            format!("{}! {}", mac.node.path, ident)
        },
    };

    let style = if FORCED_BRACKET_MACROS.contains(&&macro_name[..]) {
        MacroStyle::Brackets
    } else {
        original_style
    };

    let ts: TokenStream = mac.node.stream();
    if ts.is_empty() && !contains_comment(&context.snippet(mac.span)) {
        return match style {
            MacroStyle::Parens if position == MacroPosition::Item => {
                Some(format!("{}();", macro_name))
            }
            MacroStyle::Parens => Some(format!("{}()", macro_name)),
            MacroStyle::Brackets => Some(format!("{}[]", macro_name)),
            MacroStyle::Braces => Some(format!("{}{{}}", macro_name)),
        };
    }

    let mut parser = new_parser_from_tts(context.parse_session, ts.trees().collect());
    let mut arg_vec = Vec::new();
    let mut vec_with_semi = false;
    let mut trailing_comma = false;

    if MacroStyle::Braces != style {
        loop {
            match parse_macro_arg(&mut parser) {
                Some(arg) => arg_vec.push(arg),
                None => return Some(context.snippet(mac.span)),
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
                                None => return Some(context.snippet(mac.span)),
                            }
                        }
                    }
                    return Some(context.snippet(mac.span));
                }
                _ => return Some(context.snippet(mac.span)),
            }

            parser.bump();

            if parser.token == Token::Eof {
                trailing_comma = true;
                break;
            }
        }
    }

    match style {
        MacroStyle::Parens => {
            // Format macro invocation as function call, forcing no trailing
            // comma because not all macros support them.
            rewrite_call_inner(
                context,
                &macro_name,
                &arg_vec.iter().map(|e| &*e).collect::<Vec<_>>()[..],
                mac.span,
                shape,
                context.config.fn_call_width(),
                trailing_comma,
            ).map(|rw| match position {
                MacroPosition::Item => format!("{};", rw),
                _ => rw,
            })
        }
        MacroStyle::Brackets => {
            let mac_shape = try_opt!(shape.offset_left(macro_name.len()));
            // Handle special case: `vec![expr; expr]`
            if vec_with_semi {
                let (lbr, rbr) = if context.config.spaces_within_square_brackets() {
                    ("[ ", " ]")
                } else {
                    ("[", "]")
                };
                // 6 = `vec!` + `; `
                let total_overhead = lbr.len() + rbr.len() + 6;
                let nested_shape = mac_shape.block_indent(context.config.tab_spaces());
                let lhs = try_opt!(arg_vec[0].rewrite(context, nested_shape));
                let rhs = try_opt!(arg_vec[1].rewrite(context, nested_shape));
                if !lhs.contains('\n') && !rhs.contains('\n')
                    && lhs.len() + rhs.len() + total_overhead <= shape.width
                {
                    Some(format!("{}{}{}; {}{}", macro_name, lbr, lhs, rhs, rbr))
                } else {
                    Some(format!(
                        "{}{}\n{}{};\n{}{}\n{}{}",
                        macro_name,
                        lbr,
                        nested_shape.indent.to_string(context.config),
                        lhs,
                        nested_shape.indent.to_string(context.config),
                        rhs,
                        shape.indent.to_string(context.config),
                        rbr
                    ))
                }
            } else {
                // If we are rewriting `vec!` macro or other special macros,
                // then we can rewrite this as an usual array literal.
                // Otherwise, we must preserve the original existence of trailing comma.
                if FORCED_BRACKET_MACROS.contains(&macro_name.as_str()) {
                    context.inside_macro = false;
                    trailing_comma = false;
                }
                // Convert `MacroArg` into `ast::Expr`, as `rewrite_array` only accepts the latter.
                let expr_vec: Vec<_> = arg_vec
                    .iter()
                    .filter_map(|e| match *e {
                        MacroArg::Expr(ref e) => Some(e.clone()),
                        _ => None,
                    })
                    .collect();
                if expr_vec.len() != arg_vec.len() {
                    return Some(context.snippet(mac.span));
                }
                let sp = mk_sp(
                    context
                        .codemap
                        .span_after(mac.span, original_style.opener()),
                    mac.span.hi() - BytePos(1),
                );
                let rewrite = try_opt!(rewrite_array(
                    expr_vec.iter(),
                    sp,
                    context,
                    mac_shape,
                    trailing_comma,
                ));

                Some(format!("{}{}", macro_name, rewrite))
            }
        }
        MacroStyle::Braces => {
            // Skip macro invocations with braces, for now.
            indent_macro_snippet(context, &context.snippet(mac.span), shape.indent)
        }
    }
}

/// Tries to convert a macro use into a short hand try expression. Returns None
/// when the macro is not an instance of try! (or parsing the inner expression
/// failed).
pub fn convert_try_mac(mac: &ast::Mac, context: &RewriteContext) -> Option<ast::Expr> {
    if &format!("{}", mac.node.path)[..] == "try" {
        let ts: TokenStream = mac.node.tts.clone().into();
        let mut parser = new_parser_from_tts(context.parse_session, ts.trees().collect());

        Some(ast::Expr {
            id: ast::NodeId::new(0), // dummy value
            node: ast::ExprKind::Try(try_opt!(parser.parse_expr().ok())),
            span: mac.span, // incorrect span, but shouldn't matter too much
            attrs: ThinVec::new(),
        })
    } else {
        None
    }
}

fn macro_style(mac: &ast::Mac, context: &RewriteContext) -> MacroStyle {
    let snippet = context.snippet(mac.span);
    let paren_pos = snippet.find_uncommented("(").unwrap_or(usize::max_value());
    let bracket_pos = snippet.find_uncommented("[").unwrap_or(usize::max_value());
    let brace_pos = snippet.find_uncommented("{").unwrap_or(usize::max_value());

    if paren_pos < bracket_pos && paren_pos < brace_pos {
        MacroStyle::Parens
    } else if bracket_pos < brace_pos {
        MacroStyle::Brackets
    } else {
        MacroStyle::Braces
    }
}

/// Indent each line according to the specified `indent`.
/// e.g.
/// ```rust
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
/// will become
/// ```rust
/// foo!{
///     x,
///     y,
///     foo(
///         a,
///         b,
///         c,
//      ),
/// }
/// ```
fn indent_macro_snippet(
    context: &RewriteContext,
    macro_str: &str,
    indent: Indent,
) -> Option<String> {
    let mut lines = macro_str.lines();
    let first_line = try_opt!(lines.next().map(|s| s.trim_right()));
    let mut trimmed_lines = Vec::with_capacity(16);

    let min_prefix_space_width = try_opt!(
        lines
            .filter_map(|line| {
                let prefix_space_width = if is_empty_line(line) {
                    None
                } else {
                    Some(get_prefix_space_width(context, line))
                };
                trimmed_lines.push((line.trim(), prefix_space_width));
                prefix_space_width
            })
            .min()
    );

    Some(
        String::from(first_line) + "\n"
            + &trimmed_lines
                .iter()
                .map(|&(line, prefix_space_width)| match prefix_space_width {
                    Some(original_indent_width) => {
                        let new_indent_width = indent.width()
                            + original_indent_width
                                .checked_sub(min_prefix_space_width)
                                .unwrap_or(0);
                        let new_indent = Indent::from_width(context.config, new_indent_width);
                        format!("{}{}", new_indent.to_string(context.config), line.trim())
                    }
                    None => String::new(),
                })
                .collect::<Vec<_>>()
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
