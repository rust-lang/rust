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
use syntax::codemap::{mk_sp, BytePos};
use syntax::parse::token::Token;
use syntax::parse::tts_to_parser;
use syntax::symbol;
use syntax::util::ThinVec;

use Shape;
use codemap::SpanUtils;
use rewrite::{Rewrite, RewriteContext};
use expr::{rewrite_call, rewrite_array};
use comment::{FindUncommented, contains_comment};

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

pub fn rewrite_macro(mac: &ast::Mac,
                     extra_ident: Option<ast::Ident>,
                     context: &RewriteContext,
                     shape: Shape,
                     position: MacroPosition)
                     -> Option<String> {
    if context.config.use_try_shorthand {
        if let Some(expr) = convert_try_mac(mac, context) {
            return expr.rewrite(context, shape);
        }
    }

    let original_style = macro_style(mac, context);

    let macro_name = match extra_ident {
        None => format!("{}!", mac.node.path),
        Some(ident) => {
            if ident == symbol::keywords::Invalid.ident() {
                format!("{}!", mac.node.path)
            } else {
                format!("{}! {}", mac.node.path, ident)
            }
        }
    };

    let style = if FORCED_BRACKET_MACROS.contains(&&macro_name[..]) {
        MacroStyle::Brackets
    } else {
        original_style
    };

    if mac.node.tts.is_empty() && !contains_comment(&context.snippet(mac.span)) {
        return match style {
                   MacroStyle::Parens if position == MacroPosition::Item => {
                       Some(format!("{}();", macro_name))
                   }
                   MacroStyle::Parens => Some(format!("{}()", macro_name)),
                   MacroStyle::Brackets => Some(format!("{}[]", macro_name)),
                   MacroStyle::Braces => Some(format!("{}{{}}", macro_name)),
               };
    }

    let mut parser = tts_to_parser(context.parse_session, mac.node.tts.clone());
    let mut expr_vec = Vec::new();

    if MacroStyle::Braces != style {
        loop {
            let expr = match parser.parse_expr() {
                Ok(expr) => {
                    // Recovered errors.
                    if context.parse_session.span_diagnostic.has_errors() {
                        return None;
                    }

                    expr
                }
                Err(mut e) => {
                    e.cancel();
                    return None;
                }
            };

            expr_vec.push(expr);

            match parser.token {
                Token::Eof => break,
                Token::Comma => (),
                _ => return None,
            }

            parser.bump();

            if parser.token == Token::Eof {
                // vec! is a special case of bracket macro which should be formated as an array.
                if macro_name == "vec!" {
                    break;
                } else {
                    return None;
                }
            }
        }
    }

    match style {
        MacroStyle::Parens => {
            // Format macro invocation as function call, forcing no trailing
            // comma because not all macros support them.
            rewrite_call(context, &macro_name, &expr_vec, mac.span, shape, true)
                .map(|rw| match position {
                         MacroPosition::Item => format!("{};", rw),
                         _ => rw,
                     })
        }
        MacroStyle::Brackets => {
            // Format macro invocation as array literal.
            let extra_offset = macro_name.len();
            let shape = try_opt!(shape.shrink_left(extra_offset));
            let rewrite = try_opt!(rewrite_array(expr_vec.iter().map(|x| &**x),
                                                 mk_sp(context
                                                           .codemap
                                                           .span_after(mac.span,
                                                                       original_style
                                                                           .opener()),
                                                       mac.span.hi - BytePos(1)),
                                                 context,
                                                 shape));

            Some(format!("{}{}", macro_name, rewrite))
        }
        MacroStyle::Braces => {
            // Skip macro invocations with braces, for now.
            None
        }
    }
}

/// Tries to convert a macro use into a short hand try expression. Returns None
/// when the macro is not an instance of try! (or parsing the inner expression
/// failed).
pub fn convert_try_mac(mac: &ast::Mac, context: &RewriteContext) -> Option<ast::Expr> {
    if &format!("{}", mac.node.path)[..] == "try" {
        let mut parser = tts_to_parser(context.parse_session, mac.node.tts.clone());

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
