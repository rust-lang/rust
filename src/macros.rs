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
use syntax::parse::token::Token;
use syntax::parse::tts_to_parser;
use syntax::codemap::{mk_sp, BytePos};

use Indent;
use rewrite::RewriteContext;
use expr::{rewrite_call, rewrite_array};
use comment::FindUncommented;
use utils::{wrap_str, span_after};

static FORCED_BRACKET_MACROS: &'static [&'static str] = &["vec!"];

// FIXME: use the enum from libsyntax?
#[derive(Clone, Copy)]
enum MacroStyle {
    Parens,
    Brackets,
    Braces,
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
                     context: &RewriteContext,
                     width: usize,
                     offset: Indent)
                     -> Option<String> {
    let original_style = macro_style(mac, context);
    let macro_name = format!("{}!", mac.node.path);
    let style = if FORCED_BRACKET_MACROS.contains(&&macro_name[..]) {
        MacroStyle::Brackets
    } else {
        original_style
    };

    if let MacroStyle::Braces = style {
        return None;
    } else if mac.node.tts.is_empty() {
        return if let MacroStyle::Parens = style {
            Some(format!("{}()", macro_name))
        } else {
            Some(format!("{}[]", macro_name))
        };
    }

    let mut parser = tts_to_parser(context.parse_session, mac.node.tts.clone(), Vec::new());
    let mut expr_vec = Vec::new();

    loop {
        expr_vec.push(match parser.parse_expr() {
            Ok(expr) => expr,
            Err(..) => return None,
        });

        match parser.token {
            Token::Eof => break,
            Token::Comma => (),
            _ => return None,
        }

        let _ = parser.bump();

        if parser.token == Token::Eof {
            return None;
        }
    }

    match style {
        MacroStyle::Parens => {
            // Format macro invocation as function call.
            rewrite_call(context, &macro_name, &expr_vec, mac.span, width, offset)
        }
        MacroStyle::Brackets => {
            // Format macro invocation as array literal.
            let extra_offset = macro_name.len();
            let rewrite = try_opt!(rewrite_array(expr_vec.iter().map(|x| &**x),
                                                 mk_sp(span_after(mac.span,
                                                                  original_style.opener(),
                                                                  context.codemap),
                                                       mac.span.hi - BytePos(1)),
                                                 context,
                                                 try_opt!(width.checked_sub(extra_offset)),
                                                 offset + extra_offset));

            Some(format!("{}{}", macro_name, rewrite))
        }
        MacroStyle::Braces => {
            // Skip macro invocations with braces, for now.
            wrap_str(context.snippet(mac.span),
                     context.config.max_width,
                     width,
                     offset)
        }
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
