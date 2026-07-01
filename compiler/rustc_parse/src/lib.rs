//! The main parser interface.

// tidy-alphabetical-start
#![cfg_attr(test, feature(iter_order_by))]
#![feature(debug_closure_helpers)]
#![feature(default_field_values)]
#![feature(deref_patterns)]
#![feature(iter_intersperse)]
#![recursion_limit = "256"]
// tidy-alphabetical-end

use std::path::{Path, PathBuf};
use std::str::Utf8Error;
use std::sync::Arc;

use rustc_ast as ast;
use rustc_ast::token;
use rustc_ast::tokenstream::{TokenStream, TokenTree};
use rustc_ast_pretty::pprust;
use rustc_errors::{Diag, EmissionGuarantee, FatalError, PResult, pluralize};
pub use rustc_lexer::UNICODE_VERSION;
use rustc_session::parse::ParseSess;
use rustc_span::source_map::SourceMap;
use rustc_span::{FileName, SourceFile, Span};

pub const MACRO_ARGUMENTS: Option<&str> = Some("macro arguments");

#[macro_use]
pub mod parser;
use parser::Parser;

use crate::lexer::StripTokens;

pub mod lexer;

mod errors;

// Make sure that the Unicode version of the dependencies is the same.
const _: () = {
    let rustc_lexer = rustc_lexer::UNICODE_VERSION;
    let rustc_span = rustc_span::UNICODE_VERSION;
    let normalization = unicode_normalization::UNICODE_VERSION;
    let width = unicode_width::UNICODE_VERSION;

    if rustc_lexer.0 != rustc_span.0
        || rustc_lexer.1 != rustc_span.1
        || rustc_lexer.2 != rustc_span.2
    {
        panic!(
            "rustc_lexer and rustc_span must use the same Unicode version, \
            `rustc_lexer::UNICODE_VERSION` and `rustc_span::UNICODE_VERSION` are \
            different."
        );
    }

    if rustc_lexer.0 != normalization.0
        || rustc_lexer.1 != normalization.1
        || rustc_lexer.2 != normalization.2
    {
        panic!(
            "rustc_lexer and unicode-normalization must use the same Unicode version, \
            `rustc_lexer::UNICODE_VERSION` and `unicode_normalization::UNICODE_VERSION` are \
            different."
        );
    }

    if rustc_lexer.0 != width.0 || rustc_lexer.1 != width.1 || rustc_lexer.2 != width.2 {
        panic!(
            "rustc_lexer and unicode-width must use the same Unicode version, \
            `rustc_lexer::UNICODE_VERSION` and `unicode_width::UNICODE_VERSION` are \
            different."
        );
    }
};

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

/// Creates a new parser from a source string.
///
/// On failure, the errors must be consumed via `unwrap_or_emit_fatal`, `emit`, `cancel`,
/// etc., otherwise a panic will occur when they are dropped.
pub fn new_parser_from_source_str(
    psess: &ParseSess,
    name: FileName,
    source: String,
    strip_tokens: StripTokens,
) -> Result<Parser<'_>, Vec<Diag<'_>>> {
    let source_file = psess.source_map().new_source_file(name, source);
    new_parser_from_source_file(psess, source_file, strip_tokens)
}

/// Creates a new parser from a filename. On failure, the errors must be consumed via
/// `unwrap_or_emit_fatal`, `emit`, `cancel`, etc., otherwise a panic will occur when they are
/// dropped.
///
/// If a span is given, that is used on an error as the source of the problem.
pub fn new_parser_from_file<'a>(
    psess: &'a ParseSess,
    path: &Path,
    strip_tokens: StripTokens,
    sp: Option<Span>,
) -> Result<Parser<'a>, Vec<Diag<'a>>> {
    let sm = psess.source_map();
    let source_file = sm.load_file(path).unwrap_or_else(|e| {
        let msg = format!("couldn't read `{}`: {}", path.display(), e);
        let mut err = psess.dcx().struct_fatal(msg);
        if let Ok(contents) = std::fs::read(path)
            && let Err(utf8err) = std::str::from_utf8(&contents)
        {
            utf8_error(sm, &path.display().to_string(), sp, &mut err, utf8err, &contents);
        }
        if let Some(sp) = sp {
            err.span(sp);
        }
        err.emit()
    });
    new_parser_from_source_file(psess, source_file, strip_tokens)
}

pub fn utf8_error<E: EmissionGuarantee>(
    sm: &SourceMap,
    path: &str,
    sp: Option<Span>,
    err: &mut Diag<'_, E>,
    utf8err: Utf8Error,
    contents: &[u8],
) {
    // The file exists, but it wasn't valid UTF-8.
    let start = utf8err.valid_up_to();
    let note = format!("invalid utf-8 at byte `{start}`");
    let msg = if let Some(len) = utf8err.error_len() {
        format!(
            "byte{s} `{bytes}` {are} not valid utf-8",
            bytes = if len == 1 {
                format!("{:?}", contents[start])
            } else {
                format!("{:?}", &contents[start..start + len])
            },
            s = pluralize!(len),
            are = if len == 1 { "is" } else { "are" },
        )
    } else {
        note.clone()
    };
    let contents = String::from_utf8_lossy(contents).to_string();

    // We only emit this error for files in the current session
    // so the working directory can only be the current working directory
    let filename = FileName::Real(
        sm.path_mapping().to_real_filename(sm.working_dir(), PathBuf::from(path).as_path()),
    );
    let source = sm.new_source_file(filename, contents);

    // Avoid out-of-bounds span from lossy UTF-8 conversion.
    if start as u32 > source.normalized_source_len.0 {
        err.note(note);
        return;
    }

    let span = Span::with_root_ctxt(
        source.normalized_byte_pos(start as u32),
        source.normalized_byte_pos(start as u32),
    );
    if span.is_dummy() {
        err.note(note);
    } else {
        if sp.is_some() {
            err.span_note(span, msg);
        } else {
            err.span(span);
            err.span_label(span, msg);
        }
    }
}

/// Given a session and a `source_file`, return a parser. Returns any buffered errors from lexing
/// the initial token stream.
fn new_parser_from_source_file(
    psess: &ParseSess,
    source_file: Arc<SourceFile>,
    strip_tokens: StripTokens,
) -> Result<Parser<'_>, Vec<Diag<'_>>> {
    let end_pos = source_file.end_position();
    let stream = source_file_to_stream(psess, source_file, None, strip_tokens)?;
    let mut parser = Parser::new(psess, stream, None);
    if parser.token == token::Eof {
        parser.token.span = Span::new(end_pos, end_pos, parser.token.span.ctxt(), None);
    }
    Ok(parser)
}

/// Given a source string, produces a sequence of token trees.
///
/// NOTE: This only strips shebangs, not frontmatter!
pub fn source_str_to_stream(
    psess: &ParseSess,
    name: FileName,
    source: String,
    override_span: Option<Span>,
) -> Result<TokenStream, Vec<Diag<'_>>> {
    let source_file = psess.source_map().new_source_file(name, source);
    // FIXME(frontmatter): Consider stripping frontmatter in a future edition. We can't strip them
    // in the current edition since that would be breaking.
    // See also <https://github.com/rust-lang/rust/issues/145520>.
    // Alternatively, stop stripping shebangs here, too, if T-lang and crater approve.
    source_file_to_stream(psess, source_file, override_span, StripTokens::Shebang)
}

/// Given a source file, produces a sequence of token trees.
///
/// Returns any buffered errors from parsing the token stream.
fn source_file_to_stream<'psess>(
    psess: &'psess ParseSess,
    source_file: Arc<SourceFile>,
    override_span: Option<Span>,
    strip_tokens: StripTokens,
) -> Result<TokenStream, Vec<Diag<'psess>>> {
    let src = source_file.src.as_ref().unwrap_or_else(|| {
        psess.dcx().bug(format!(
            "cannot lex `source_file` without source: {}",
            psess.source_map().filename_for_diagnostics(&source_file.name)
        ));
    });

    lexer::lex_token_trees(psess, src.as_str(), source_file.start_pos, override_span, strip_tokens)
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

pub fn fake_token_stream_for_item(
    psess: &ParseSess,
    item: &ast::Item,
    attr_to_exclude: Option<&ast::Attribute>,
) -> TokenStream {
    if let Some(tokens) = fake_token_stream_for_file_mod(psess, item, attr_to_exclude) {
        return tokens;
    }

    let source = pprust::item_to_string(item);
    let filename = FileName::macro_expansion_source_code(&source);
    unwrap_or_emit_fatal(source_str_to_stream(psess, filename, source, Some(item.span)))
}

fn fake_token_stream_for_file_mod(
    psess: &ParseSess,
    item: &ast::Item,
    attr_to_exclude: Option<&ast::Attribute>,
) -> Option<TokenStream> {
    let ast::ItemKind::Mod(safety, ident, ast::ModKind::Loaded(_, ast::Inline::No { .. }, spans)) =
        &item.kind
    else {
        return None;
    };

    let attrs_hi = item
        .attrs
        .iter()
        .filter(|attr| attr.style == ast::AttrStyle::Inner)
        .map(|attr| attr.span.hi())
        .chain(
            attr_to_exclude
                .filter(|attr| attr.style == ast::AttrStyle::Inner)
                .map(|attr| attr.span.hi()),
        )
        .max()
        .unwrap_or(spans.inject_use_span.lo());
    let body_span = spans.inner_span.with_lo(attrs_hi);
    let body_src = psess.source_map().span_to_snippet(body_span).ok()?;
    let body_stream = unwrap_or_emit_fatal(lexer::lex_token_trees(
        psess,
        &body_src,
        body_span.lo(),
        None,
        StripTokens::Nothing,
    ));

    let mut body_tts = Vec::new();
    for attr in item.attrs.iter().filter(|attr| attr.style == ast::AttrStyle::Inner) {
        body_tts.extend(attr.token_trees());
    }
    body_tts.extend(body_stream.iter().cloned());

    // Synthesize only the `mod name { ... }` wrapper. The body tokens come from
    // the loaded file so diagnostics inside the module keep their real spans.
    let wrapper = ast::Item {
        attrs: item
            .attrs
            .iter()
            .filter(|attr| attr.style == ast::AttrStyle::Outer)
            .cloned()
            .collect(),
        id: item.id,
        span: item.span,
        vis: item.vis.clone(),
        kind: ast::ItemKind::Mod(
            *safety,
            *ident,
            ast::ModKind::Loaded(Default::default(), ast::Inline::Yes, ast::ModSpans::default()),
        ),
        tokens: None,
    };

    let source = pprust::item_to_string(&wrapper);
    let filename = FileName::macro_expansion_source_code(&source);
    let wrapper_stream =
        unwrap_or_emit_fatal(source_str_to_stream(psess, filename, source, Some(item.span)));
    let mut wrapper_tts: Vec<_> = wrapper_stream.iter().cloned().collect();

    let Some(TokenTree::Delimited(_, _, token::Delimiter::Brace, stream)) = wrapper_tts
        .iter_mut()
        .rev()
        .find(|tt| matches!(tt, TokenTree::Delimited(_, _, token::Delimiter::Brace, _)))
    else {
        return None;
    };
    *stream = TokenStream::new(body_tts);

    Some(TokenStream::new(wrapper_tts))
}

pub fn fake_token_stream_for_foreign_item(
    psess: &ParseSess,
    item: &ast::ForeignItem,
) -> TokenStream {
    let source = pprust::foreign_item_to_string(item);
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
