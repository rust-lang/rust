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

use std::thread;

use syntax::ast;
use syntax::parse::token::{Eof, Comma, Token};
use syntax::parse::{ParseSess, tts_to_parser};

use Indent;
use rewrite::RewriteContext;
use expr::{rewrite_call, rewrite_array};
use comment::FindUncommented;
use utils::wrap_str;

// We need to pass `TokenTree`s to our expression parsing thread, but they are
// not `Send`. We wrap them in a `Send` container to force our will.
// FIXME: this is a pretty terrible hack. Any other solution would be preferred.
struct ForceSend<T>(pub T);
unsafe impl<T> Send for ForceSend<T> {}

// FIXME: use the enum from libsyntax?
enum MacroStyle {
    Parens,
    Brackets,
    Braces,
}

pub fn rewrite_macro(mac: &ast::Mac,
                     context: &RewriteContext,
                     width: usize,
                     offset: Indent)
                     -> Option<String> {
    let ast::Mac_::MacInvocTT(ref path, ref tt_vec, _) = mac.node;
    let style = macro_style(mac, context);
    let macro_name = format!("{}!", path);

    if let MacroStyle::Braces = style {
        return None;
    } else if tt_vec.is_empty() {
        return if let MacroStyle::Parens = style {
            Some(format!("{}()", macro_name))
        } else {
            Some(format!("{}[]", macro_name))
        };
    }

    let wrapped_tt_vec = ForceSend((*tt_vec).clone());
    // Wrap expression parsing logic in a thread since the libsyntax parser
    // panicks on failure, which we do not want to propagate.
    let expr_vec_result = thread::catch_panic(move || {
        let parse_session = ParseSess::new();
        let mut parser = tts_to_parser(&parse_session, wrapped_tt_vec.0, vec![]);
        let mut expr_vec = vec![];

        loop {
            expr_vec.push(parser.parse_expr());

            match parser.token {
                Token::Eof => break,
                Token::Comma => (),
                _ => panic!("Macro not list-like, skiping..."),
            }

            let _ = parser.bump();
        }

        expr_vec
    });
    let expr_vec = try_opt!(expr_vec_result.ok());

    match style {
        MacroStyle::Parens => {
            // Format macro invocation as function call.
            rewrite_call(context, &macro_name, &expr_vec, mac.span, width, offset)
        }
        MacroStyle::Brackets => {
            // Format macro invocation as array literal.
            let extra_offset = macro_name.len();
            let rewrite = try_opt!(rewrite_array(expr_vec.iter().map(|x| &**x),
                                                 mac.span,
                                                 context,
                                                 try_opt!(width.checked_sub(extra_offset)),
                                                 offset + extra_offset));
            Some(format!("{}{}", macro_name, rewrite))
        }
        MacroStyle::Braces => {
            // Skip macro invocations with braces, for now.
            wrap_str(context.snippet(mac.span), context.config.max_width, width, offset)
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
