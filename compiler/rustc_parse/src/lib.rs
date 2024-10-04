//! The main parser interface.

// tidy-alphabetical-start
#![allow(internal_features)]
#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]
#![feature(array_windows)]
#![feature(assert_matches)]
#![feature(box_patterns)]
#![feature(debug_closure_helpers)]
#![feature(if_let_guard)]
#![feature(iter_intersperse)]
#![feature(let_chains)]
#![warn(unreachable_pub)]
// tidy-alphabetical-end

use std::path::Path;

use rustc_ast as ast;
use rustc_ast::tokenstream::TokenStream;
use rustc_ast::{AttrItem, Attribute, NestedMetaItem, token};
use rustc_ast_pretty::pprust;
use rustc_data_structures::sync::Lrc;
use rustc_errors::{Diag, FatalError, PResult};
use rustc_session::parse::ParseSess;
use rustc_span::{FileName, SourceFile, Span};

pub const MACRO_ARGUMENTS: Option<&str> = Some("macro arguments");

#[macro_use]
pub mod parser;
use parser::{Parser, make_unclosed_delims_error};
pub mod lexer;
pub mod validate_attr;

mod errors;

rustc_fluent_macro::fluent_messages! { "../messages.ftl" }

// Unwrap the result if `Ok`, otherwise emit the diagnostics and abort.
pub fn unwrap_or_emit_fatal<T>(expr: Result<T, Vec<Diag<'_>>>) -> T {
    match expr {
        Ok(expr) => expr,
        Err(errs) => {
            for err in errs {
                err.emit();
            }
            FatalError.raise()
        }
    }
}

/// Creates a new parser from a source string. On failure, the errors must be consumed via
/// `unwrap_or_emit_fatal`, `emit`, `cancel`, etc., otherwise a panic will occur when they are
/// dropped.
pub fn new_parser_from_source_str(
    psess: &ParseSess,
    name: FileName,
    source: String,
) -> Result<Parser<'_>, Vec<Diag<'_>>> {
    let source_file = psess.source_map().new_source_file(name, source);
    new_parser_from_source_file(psess, source_file)
}

/// Creates a new parser from a filename. On failure, the errors must be consumed via
/// `unwrap_or_emit_fatal`, `emit`, `cancel`, etc., otherwise a panic will occur when they are
/// dropped.
///
/// If a span is given, that is used on an error as the source of the problem.
pub fn new_parser_from_file<'a>(
    psess: &'a ParseSess,
    path: &Path,
    sp: Option<Span>,
) -> Result<Parser<'a>, Vec<Diag<'a>>> {
    let source_file = psess.source_map().load_file(path).unwrap_or_else(|e| {
        let msg = format!("couldn't read {}: {}", path.display(), e);
        let mut err = psess.dcx().struct_fatal(msg);
        if let Some(sp) = sp {
            err.span(sp);
        }
        err.emit();
    });
    new_parser_from_source_file(psess, source_file)
}

/// Given a session and a `source_file`, return a parser. Returns any buffered errors from lexing
/// the initial token stream.
fn new_parser_from_source_file(
    psess: &ParseSess,
    source_file: Lrc<SourceFile>,
) -> Result<Parser<'_>, Vec<Diag<'_>>> {
    let end_pos = source_file.end_position();
    let stream = source_file_to_stream(psess, source_file, None)?;
    let mut parser = Parser::new(psess, stream, None);
    if parser.token == token::Eof {
        parser.token.span = Span::new(end_pos, end_pos, parser.token.span.ctxt(), None);
    }
    Ok(parser)
}

pub fn source_str_to_stream(
    psess: &ParseSess,
    name: FileName,
    source: String,
    override_span: Option<Span>,
) -> Result<TokenStream, Vec<Diag<'_>>> {
    let source_file = psess.source_map().new_source_file(name, source);
    source_file_to_stream(psess, source_file, override_span)
}

/// Given a source file, produces a sequence of token trees. Returns any buffered errors from
/// parsing the token stream.
fn source_file_to_stream<'psess>(
    psess: &'psess ParseSess,
    source_file: Lrc<SourceFile>,
    override_span: Option<Span>,
) -> Result<TokenStream, Vec<Diag<'psess>>> {
    let src = source_file.src.as_ref().unwrap_or_else(|| {
        psess.dcx().bug(format!(
            "cannot lex `source_file` without source: {}",
            psess.source_map().filename_for_diagnostics(&source_file.name)
        ));
    });

    lexer::lex_token_trees(psess, src.as_str(), source_file.start_pos, override_span)
}

/// Runs the given subparser `f` on the tokens of the given `attr`'s item.
pub fn parse_in<'a, T>(
    psess: &'a ParseSess,
    tts: TokenStream,
    name: &'static str,
    mut f: impl FnMut(&mut Parser<'a>) -> PResult<'a, T>,
) -> PResult<'a, T> {
    let mut parser = Parser::new(psess, tts, Some(name));
    let result = f(&mut parser)?;
    if parser.token != token::Eof {
        parser.unexpected()?;
    }
    Ok(result)
}

pub fn fake_token_stream_for_item(psess: &ParseSess, item: &ast::Item) -> TokenStream {
    let source = pprust::item_to_string(item);
    let filename = FileName::macro_expansion_source_code(&source);
    unwrap_or_emit_fatal(source_str_to_stream(psess, filename, source, Some(item.span)))
}

pub fn fake_token_stream_for_crate(psess: &ParseSess, krate: &ast::Crate) -> TokenStream {
    let source = pprust::crate_to_string_for_macros(krate);
    let filename = FileName::macro_expansion_source_code(&source);
    unwrap_or_emit_fatal(source_str_to_stream(
        psess,
        filename,
        source,
        Some(krate.spans.inner_span),
    ))
}

pub fn parse_cfg_attr(
    cfg_attr: &Attribute,
    psess: &ParseSess,
) -> Option<(NestedMetaItem, Vec<(AttrItem, Span)>)> {
    const CFG_ATTR_GRAMMAR_HELP: &str = "#[cfg_attr(condition, attribute, other_attribute, ...)]";
    const CFG_ATTR_NOTE_REF: &str = "for more information, visit \
        <https://doc.rust-lang.org/reference/conditional-compilation.html#the-cfg_attr-attribute>";

    match cfg_attr.get_normal_item().args {
        ast::AttrArgs::Delimited(ast::DelimArgs { dspan, delim, ref tokens })
            if !tokens.is_empty() =>
        {
            crate::validate_attr::check_cfg_attr_bad_delim(psess, dspan, delim);
            match parse_in(psess, tokens.clone(), "`cfg_attr` input", |p| p.parse_cfg_attr()) {
                Ok(r) => return Some(r),
                Err(e) => {
                    e.with_help(format!("the valid syntax is `{CFG_ATTR_GRAMMAR_HELP}`"))
                        .with_note(CFG_ATTR_NOTE_REF)
                        .emit();
                }
            }
        }
        _ => {
            psess.dcx().emit_err(errors::MalformedCfgAttr {
                span: cfg_attr.span,
                sugg: CFG_ATTR_GRAMMAR_HELP,
            });
        }
    }
    None
}
