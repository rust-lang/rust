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
use syntax::ast;
use syntax::codemap::{BytePos, Span};
use syntax::parse::new_parser_from_tts;
use syntax::parse::parser::Parser;
use syntax::parse::token::{BinOpToken, DelimToken, Token};
use syntax::print::pprust;
use syntax::symbol;
use syntax::tokenstream::{Cursor, ThinTokenStream, TokenStream, TokenTree};
use syntax::util::ThinVec;

use codemap::SpanUtils;
use comment::{contains_comment, remove_trailing_white_spaces, FindUncommented};
use expr::{rewrite_array, rewrite_call_inner};
use rewrite::{Rewrite, RewriteContext};
use shape::{Indent, Shape};
use utils::{format_visibility, mk_sp};

const FORCED_BRACKET_MACROS: &[&str] = &["vec!"];

// FIXME: use the enum from libsyntax?
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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
    Pat,
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

#[derive(Debug)]
pub enum MacroArg {
    Expr(ast::Expr),
    Ty(ast::Ty),
    Pat(ast::Pat),
}

impl Rewrite for MacroArg {
    fn rewrite(&self, context: &RewriteContext, shape: Shape) -> Option<String> {
        match *self {
            MacroArg::Expr(ref expr) => expr.rewrite(context, shape),
            MacroArg::Ty(ref ty) => ty.rewrite(context, shape),
            MacroArg::Pat(ref pat) => pat.rewrite(context, shape),
        }
    }
}

fn parse_macro_arg(parser: &mut Parser) -> Option<MacroArg> {
    macro_rules! parse_macro_arg {
        ($macro_arg: ident, $parser: ident) => {
            let mut cloned_parser = (*parser).clone();
            match cloned_parser.$parser() {
                Ok(x) => {
                    if parser.sess.span_diagnostic.has_errors() {
                        parser.sess.span_diagnostic.reset_err_count();
                    } else {
                        // Parsing succeeded.
                        *parser = cloned_parser;
                        return Some(MacroArg::$macro_arg((*x).clone()));
                    }
                }
                Err(mut e) => {
                    e.cancel();
                    parser.sess.span_diagnostic.reset_err_count();
                }
            }
        }
    }

    parse_macro_arg!(Expr, parse_expr);
    parse_macro_arg!(Ty, parse_ty);
    parse_macro_arg!(Pat, parse_pat);

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
            context.inside_macro = false;
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

    let ts: TokenStream = mac.node.stream();
    if ts.is_empty() && !contains_comment(context.snippet(mac.span)) {
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
                None => return Some(context.snippet(mac.span).to_owned()),
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
                                None => return Some(context.snippet(mac.span).to_owned()),
                            }
                        }
                    }
                    return Some(context.snippet(mac.span).to_owned());
                }
                _ => return Some(context.snippet(mac.span).to_owned()),
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
                context.config.width_heuristics().fn_call_width,
                trailing_comma,
            ).map(|rw| match position {
                MacroPosition::Item => format!("{};", rw),
                _ => rw,
            })
        }
        MacroStyle::Brackets => {
            let mac_shape = shape.offset_left(macro_name.len())?;
            // Handle special case: `vec![expr; expr]`
            if vec_with_semi {
                let (lbr, rbr) = if context.config.spaces_within_parens_and_brackets() {
                    ("[ ", " ]")
                } else {
                    ("[", "]")
                };
                // 6 = `vec!` + `; `
                let total_overhead = lbr.len() + rbr.len() + 6;
                let nested_shape = mac_shape.block_indent(context.config.tab_spaces());
                let lhs = arg_vec[0].rewrite(context, nested_shape)?;
                let rhs = arg_vec[1].rewrite(context, nested_shape)?;
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
                let sp = mk_sp(
                    context
                        .codemap
                        .span_after(mac.span, original_style.opener()),
                    mac.span.hi() - BytePos(1),
                );
                let arg_vec = &arg_vec.iter().map(|e| &*e).collect::<Vec<_>>()[..];
                let rewrite = rewrite_array(arg_vec, sp, context, mac_shape, trailing_comma)?;

                Some(format!("{}{}", macro_name, rewrite))
            }
        }
        MacroStyle::Braces => {
            // Skip macro invocations with braces, for now.
            indent_macro_snippet(context, context.snippet(mac.span), shape.indent)
        }
    }
}

pub fn rewrite_macro_def(
    context: &RewriteContext,
    indent: Indent,
    def: &ast::MacroDef,
    ident: ast::Ident,
    vis: &ast::Visibility,
    span: Span,
) -> Option<String> {
    let snippet = Some(remove_trailing_white_spaces(context.snippet(span)));

    if def.legacy {
        return snippet;
    }

    let mut parser = MacroParser::new(def.stream().into_trees());
    let mut parsed_def = match parser.parse() {
        Some(def) => def,
        None => return snippet,
    };

    // Only attempt to format function-like macros.
    if parsed_def.branches.len() != 1 || parsed_def.branches[0].args_paren_kind != DelimToken::Paren
    {
        // FIXME(#1539): implement for non-sugared macros.
        return snippet;
    }

    let branch = parsed_def.branches.remove(0);
    let args_str = format_macro_args(branch.args)?;

    // The macro body is the most interesting part. It might end up as various
    // AST nodes, but also has special variables (e.g, `$foo`) which can't be
    // parsed as regular Rust code (and note that these can be escaped using
    // `$$`). We'll try and format like an AST node, but we'll substitute
    // variables for new names with the same length first.

    let old_body = context.snippet(branch.body).trim();
    let (body_str, substs) = replace_names(old_body);

    // We'll hack the indent below, take this into account when formatting,
    let mut config = context.config.clone();
    let new_width = config.max_width() - indent.block_indent(&config).width();
    config.set().max_width(new_width);
    config.set().hide_parse_errors(true);

    // First try to format as items, then as statements.
    let new_body = match ::format_snippet(&body_str, &config) {
        Some(new_body) => new_body,
        None => match ::format_code_block(&body_str, &config) {
            Some(new_body) => new_body,
            None => return snippet,
        },
    };

    // Indent the body since it is in a block.
    let indent_str = indent.block_indent(&config).to_string(&config);
    let mut new_body = new_body
        .lines()
        .map(|l| {
            if l.is_empty() {
                l.to_owned()
            } else {
                format!("{}{}", indent_str, l)
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    // Undo our replacement of macro variables.
    // FIXME: this could be *much* more efficient.
    for (old, new) in substs.iter() {
        if old_body.find(new).is_some() {
            debug!(
                "rewrite_macro_def: bailing matching variable: `{}` in `{}`",
                new, ident
            );
            return snippet;
        }
        new_body = new_body.replace(new, old);
    }

    let result = format!(
        "{}macro {}({}) {{\n{}\n{}}}",
        format_visibility(vis),
        ident,
        args_str,
        new_body,
        indent.to_string(&context.config),
    );

    Some(result)
}

// Replaces `$foo` with `zfoo`. We must check for name overlap to ensure we
// aren't causing problems.
// This should also work for escaped `$` variables, where we leave earlier `$`s.
fn replace_names(input: &str) -> (String, HashMap<String, String>) {
    // Each substitution will require five or six extra bytes.
    let mut result = String::with_capacity(input.len() + 64);
    let mut substs = HashMap::new();
    let mut dollar_count = 0;
    let mut cur_name = String::new();

    for c in input.chars() {
        if c == '$' {
            dollar_count += 1;
        } else if dollar_count == 0 {
            result.push(c);
        } else if !c.is_alphanumeric() && !cur_name.is_empty() {
            // Terminates a name following one or more dollars.
            let mut new_name = String::new();
            let mut old_name = String::new();
            old_name.push('$');
            for _ in 0..(dollar_count - 1) {
                new_name.push('$');
                old_name.push('$');
            }
            new_name.push('z');
            new_name.push_str(&cur_name);
            old_name.push_str(&cur_name);

            result.push_str(&new_name);
            substs.insert(old_name, new_name);

            result.push(c);

            dollar_count = 0;
            cur_name = String::new();
        } else if c.is_alphanumeric() {
            cur_name.push(c);
        }
    }

    // FIXME: duplicate code
    if !cur_name.is_empty() {
        let mut new_name = String::new();
        let mut old_name = String::new();
        old_name.push('$');
        for _ in 0..(dollar_count - 1) {
            new_name.push('$');
            old_name.push('$');
        }
        new_name.push('z');
        new_name.push_str(&cur_name);
        old_name.push_str(&cur_name);

        result.push_str(&new_name);
        substs.insert(old_name, new_name);
    }

    debug!("replace_names `{}` {:?}", result, substs);

    (result, substs)
}

// This is a bit sketchy. The token rules probably need tweaking, but it works
// for some common cases. I hope the basic logic is sufficient. Note that the
// meaning of some tokens is a bit different here from usual Rust, e.g., `*`
// and `(`/`)` have special meaning.
//
// We always try and format on one line.
fn format_macro_args(toks: ThinTokenStream) -> Option<String> {
    let mut result = String::with_capacity(128);
    let mut insert_space = SpaceState::Never;

    for tok in (toks.into(): TokenStream).trees() {
        match tok {
            TokenTree::Token(_, t) => {
                if !result.is_empty() && force_space_before(&t) {
                    insert_space = SpaceState::Always;
                }
                if force_no_space_before(&t) {
                    insert_space = SpaceState::Never;
                }
                match (insert_space, ident_like(&t)) {
                    (SpaceState::Always, _)
                    | (SpaceState::Punctuation, false)
                    | (SpaceState::Ident, true) => {
                        result.push(' ');
                    }
                    _ => {}
                }
                result.push_str(&pprust::token_to_string(&t));
                insert_space = next_space(&t);
            }
            TokenTree::Delimited(_, d) => {
                let formatted = format_macro_args(d.tts)?;
                match insert_space {
                    SpaceState::Always => {
                        result.push(' ');
                    }
                    _ => {}
                }
                match d.delim {
                    DelimToken::Paren => {
                        result.push_str(&format!("({})", formatted));
                        insert_space = SpaceState::Always;
                    }
                    DelimToken::Bracket => {
                        result.push_str(&format!("[{}]", formatted));
                        insert_space = SpaceState::Always;
                    }
                    DelimToken::Brace => {
                        result.push_str(&format!(" {{ {} }}", formatted));
                        insert_space = SpaceState::Always;
                    }
                    DelimToken::NoDelim => {
                        result.push_str(&format!("{}", formatted));
                        insert_space = SpaceState::Always;
                    }
                }
            }
        }
    }

    Some(result)
}

// We should insert a space if the next token is a:
#[derive(Copy, Clone)]
enum SpaceState {
    Never,
    Punctuation,
    Ident, // Or ident/literal-like thing.
    Always,
}

fn force_space_before(tok: &Token) -> bool {
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
        | Token::Pound
        | Token::Dollar => true,
        Token::BinOp(bot) => bot != BinOpToken::Star,
        _ => false,
    }
}

fn force_no_space_before(tok: &Token) -> bool {
    match *tok {
        Token::Semi | Token::Comma | Token::Dot => true,
        Token::BinOp(bot) => bot == BinOpToken::Star,
        _ => false,
    }
}
fn ident_like(tok: &Token) -> bool {
    match *tok {
        Token::Ident(_) | Token::Literal(..) | Token::Lifetime(_) => true,
        _ => false,
    }
}

fn next_space(tok: &Token) -> SpaceState {
    match *tok {
        Token::Not
        | Token::Tilde
        | Token::At
        | Token::Comma
        | Token::Dot
        | Token::DotDot
        | Token::DotDotDot
        | Token::DotDotEq
        | Token::DotEq
        | Token::Question
        | Token::Underscore
        | Token::BinOp(_) => SpaceState::Punctuation,

        Token::ModSep
        | Token::Pound
        | Token::Dollar
        | Token::OpenDelim(_)
        | Token::CloseDelim(_)
        | Token::Whitespace => SpaceState::Never,

        Token::Literal(..) | Token::Ident(_) | Token::Lifetime(_) => SpaceState::Ident,

        _ => SpaceState::Always,
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
            node: ast::ExprKind::Try(parser.parse_expr().ok()?),
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
    let first_line = lines.next().map(|s| s.trim_right())?;
    let mut trimmed_lines = Vec::with_capacity(16);

    let min_prefix_space_width = lines
        .filter_map(|line| {
            let prefix_space_width = if is_empty_line(line) {
                None
            } else {
                Some(get_prefix_space_width(context, line))
            };
            trimmed_lines.push((line.trim(), prefix_space_width));
            prefix_space_width
        })
        .min()?;

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
        let (args_paren_kind, args) = match self.toks.next()? {
            TokenTree::Token(..) => return None,
            TokenTree::Delimited(_, ref d) => (d.delim, d.tts.clone().into()),
        };
        match self.toks.next()? {
            TokenTree::Token(_, Token::FatArrow) => {}
            _ => return None,
        }
        let body = match self.toks.next()? {
            TokenTree::Token(..) => return None,
            TokenTree::Delimited(sp, _) => {
                let data = sp.data();
                Span::new(data.lo + BytePos(1), data.hi - BytePos(1), data.ctxt)
            }
        };
        Some(MacroBranch {
            args,
            args_paren_kind,
            body,
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
    args: ThinTokenStream,
    args_paren_kind: DelimToken,
    body: Span,
}

#[cfg(test)]
mod test {
    use super::*;
    use syntax::parse::{parse_stream_from_source_str, ParseSess};
    use syntax::codemap::{FileName, FilePathMapping};

    fn format_macro_args_str(s: &str) -> String {
        let input = parse_stream_from_source_str(
            FileName::Custom("stdin".to_owned()),
            s.to_owned(),
            &ParseSess::new(FilePathMapping::empty()),
            None,
        );
        format_macro_args(input.into()).unwrap()
    }

    #[test]
    fn test_format_macro_args() {
        assert_eq!(format_macro_args_str(""), "".to_owned());
        assert_eq!(format_macro_args_str("$ x : ident"), "$x: ident".to_owned());
        assert_eq!(
            format_macro_args_str("$ m1 : ident , $ m2 : ident , $ x : ident"),
            "$m1: ident, $m2: ident, $x: ident".to_owned()
        );
        assert_eq!(
            format_macro_args_str("$($beginning:ident),*;$middle:ident;$($end:ident),*"),
            "$($beginning: ident),*; $middle: ident; $($end: ident),*".to_owned()
        );
        assert_eq!(
            format_macro_args_str(
                "$ name : ident ( $ ( $ dol : tt $ var : ident ) * ) $ ( $ body : tt ) *"
            ),
            "$name: ident($($dol: tt $var: ident)*) $($body: tt)*".to_owned()
        );
    }
}
