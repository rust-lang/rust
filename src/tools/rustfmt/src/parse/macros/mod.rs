use rustc_ast::token::{Delimiter, NonterminalKind, NtExprKind::*, NtPatKind::*, TokenKind};
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{ast, ptr};
use rustc_parse::MACRO_ARGUMENTS;
use rustc_parse::parser::{ForceCollect, Parser, Recovery};
use rustc_session::parse::ParseSess;
use rustc_span::Symbol;
use rustc_span::symbol::{self, kw};

use crate::macros::MacroArg;
use crate::rewrite::RewriteContext;

pub(crate) mod asm;
pub(crate) mod cfg_if;
pub(crate) mod lazy_static;

fn build_stream_parser<'a>(psess: &'a ParseSess, tokens: TokenStream) -> Parser<'a> {
    Parser::new(psess, tokens, MACRO_ARGUMENTS).recovery(Recovery::Forbidden)
}

fn build_parser<'a>(context: &RewriteContext<'a>, tokens: TokenStream) -> Parser<'a> {
    build_stream_parser(context.psess.inner(), tokens)
}

fn parse_macro_arg<'a, 'b: 'a>(parser: &'a mut Parser<'b>) -> Option<MacroArg> {
    macro_rules! parse_macro_arg {
        ($macro_arg:ident, $nt_kind:expr, $try_parse:expr, $then:expr) => {
            let mut cloned_parser = (*parser).clone();
            if Parser::nonterminal_may_begin_with($nt_kind, &cloned_parser.token) {
                match $try_parse(&mut cloned_parser) {
                    Ok(x) => {
                        if parser.psess.dcx().has_errors().is_some() {
                            parser.psess.dcx().reset_err_count();
                        } else {
                            // Parsing succeeded.
                            *parser = cloned_parser;
                            return Some(MacroArg::$macro_arg($then(x)?));
                        }
                    }
                    Err(e) => {
                        e.cancel();
                        parser.psess.dcx().reset_err_count();
                    }
                }
            }
        };
    }

    parse_macro_arg!(
        Expr,
        NonterminalKind::Expr(Expr),
        |parser: &mut Parser<'b>| parser.parse_expr(),
        |x: ptr::P<ast::Expr>| Some(x)
    );
    parse_macro_arg!(
        Ty,
        NonterminalKind::Ty,
        |parser: &mut Parser<'b>| parser.parse_ty(),
        |x: ptr::P<ast::Ty>| Some(x)
    );
    parse_macro_arg!(
        Pat,
        NonterminalKind::Pat(PatParam { inferred: false }),
        |parser: &mut Parser<'b>| parser.parse_pat_no_top_alt(None, None),
        |x: ptr::P<ast::Pat>| Some(x)
    );
    // `parse_item` returns `Option<ptr::P<ast::Item>>`.
    parse_macro_arg!(
        Item,
        NonterminalKind::Item,
        |parser: &mut Parser<'b>| parser.parse_item(ForceCollect::No),
        |x: Option<ptr::P<ast::Item>>| x
    );

    None
}

pub(crate) struct ParsedMacroArgs {
    pub(crate) vec_with_semi: bool,
    pub(crate) trailing_comma: bool,
    pub(crate) args: Vec<MacroArg>,
}

fn check_keyword<'a, 'b: 'a>(parser: &'a mut Parser<'b>) -> Option<MacroArg> {
    for &keyword in RUST_KW.iter() {
        if parser.token.is_keyword(keyword)
            && parser.look_ahead(1, |t| *t == TokenKind::Eof || *t == TokenKind::Comma)
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

pub(crate) fn parse_macro_args(
    context: &RewriteContext<'_>,
    tokens: TokenStream,
    style: Delimiter,
    forced_bracket: bool,
) -> Option<ParsedMacroArgs> {
    let mut parser = build_parser(context, tokens);
    let mut args = Vec::new();
    let mut vec_with_semi = false;
    let mut trailing_comma = false;

    if Delimiter::Brace != style {
        loop {
            if let Some(arg) = check_keyword(&mut parser) {
                args.push(arg);
            } else if let Some(arg) = parse_macro_arg(&mut parser) {
                args.push(arg);
            } else {
                return None;
            }

            match parser.token.kind {
                TokenKind::Eof => break,
                TokenKind::Comma => (),
                TokenKind::Semi => {
                    // Try to parse `vec![expr; expr]`
                    if forced_bracket {
                        parser.bump();
                        if parser.token.kind != TokenKind::Eof {
                            match parse_macro_arg(&mut parser) {
                                Some(arg) => {
                                    args.push(arg);
                                    parser.bump();
                                    if parser.token == TokenKind::Eof && args.len() == 2 {
                                        vec_with_semi = true;
                                        break;
                                    }
                                }
                                None => {
                                    return None;
                                }
                            }
                        }
                    }
                    return None;
                }
                _ if args.last().map_or(false, MacroArg::is_item) => continue,
                _ => return None,
            }

            parser.bump();

            if parser.token == TokenKind::Eof {
                trailing_comma = true;
                break;
            }
        }
    }

    Some(ParsedMacroArgs {
        vec_with_semi,
        trailing_comma,
        args,
    })
}

pub(crate) fn parse_expr(
    context: &RewriteContext<'_>,
    tokens: TokenStream,
) -> Option<ptr::P<ast::Expr>> {
    let mut parser = build_parser(context, tokens);
    parser.parse_expr().ok()
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
