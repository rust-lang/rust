//! The main parser interface.

#![feature(array_windows)]
#![feature(crate_visibility_modifier)]
#![feature(if_let_guard)]
#![feature(box_patterns)]
#![recursion_limit = "256"]

#[macro_use]
extern crate tracing;

use rustc_ast as ast;
use rustc_ast::token::{self, Nonterminal, Token, TokenKind};
use rustc_ast::tokenstream::{self, AttributesData, CanSynthesizeMissingTokens, LazyTokenStream};
use rustc_ast::tokenstream::{AttrAnnotatedTokenStream, AttrAnnotatedTokenTree};
use rustc_ast::tokenstream::{Spacing, TokenStream};
use rustc_ast::AstLike;
use rustc_ast::Attribute;
use rustc_ast::{AttrItem, MetaItem};
use rustc_ast_pretty::pprust;
use rustc_data_structures::sync::Lrc;
use rustc_errors::{Applicability, Diagnostic, FatalError, Level, PResult};
use rustc_session::parse::ParseSess;
use rustc_span::{FileName, SourceFile, Span};

use std::path::Path;
use std::str;

pub const MACRO_ARGUMENTS: Option<&str> = Some("macro arguments");

#[macro_use]
pub mod parser;
use parser::{emit_unclosed_delims, make_unclosed_delims_error, Parser};
pub mod lexer;
pub mod validate_attr;

// A bunch of utility functions of the form `parse_<thing>_from_<source>`
// where <thing> includes crate, expr, item, stmt, tts, and one that
// uses a HOF to parse anything, and <source> includes file and
// `source_str`.

/// A variant of 'panictry!' that works on a Vec<Diagnostic> instead of a single DiagnosticBuilder.
macro_rules! panictry_buffer {
    ($handler:expr, $e:expr) => {{
        use rustc_errors::FatalError;
        use std::result::Result::{Err, Ok};
        match $e {
            Ok(e) => e,
            Err(errs) => {
                for e in errs {
                    $handler.emit_diagnostic(&e);
                }
                FatalError.raise()
            }
        }
    }};
}

pub fn parse_crate_from_file<'a>(input: &Path, sess: &'a ParseSess) -> PResult<'a, ast::Crate> {
    let mut parser = new_parser_from_file(sess, input, None);
    parser.parse_crate_mod()
}

pub fn parse_crate_attrs_from_file<'a>(
    input: &Path,
    sess: &'a ParseSess,
) -> PResult<'a, Vec<ast::Attribute>> {
    let mut parser = new_parser_from_file(sess, input, None);
    parser.parse_inner_attributes()
}

pub fn parse_crate_from_source_str(
    name: FileName,
    source: String,
    sess: &ParseSess,
) -> PResult<'_, ast::Crate> {
    new_parser_from_source_str(sess, name, source).parse_crate_mod()
}

pub fn parse_crate_attrs_from_source_str(
    name: FileName,
    source: String,
    sess: &ParseSess,
) -> PResult<'_, Vec<ast::Attribute>> {
    new_parser_from_source_str(sess, name, source).parse_inner_attributes()
}

pub fn parse_stream_from_source_str(
    name: FileName,
    source: String,
    sess: &ParseSess,
    override_span: Option<Span>,
) -> TokenStream {
    let (stream, mut errors) =
        source_file_to_stream(sess, sess.source_map().new_source_file(name, source), override_span);
    emit_unclosed_delims(&mut errors, &sess);
    stream
}

/// Creates a new parser from a source string.
pub fn new_parser_from_source_str(sess: &ParseSess, name: FileName, source: String) -> Parser<'_> {
    panictry_buffer!(&sess.span_diagnostic, maybe_new_parser_from_source_str(sess, name, source))
}

/// Creates a new parser from a source string. Returns any buffered errors from lexing the initial
/// token stream.
pub fn maybe_new_parser_from_source_str(
    sess: &ParseSess,
    name: FileName,
    source: String,
) -> Result<Parser<'_>, Vec<Diagnostic>> {
    maybe_source_file_to_parser(sess, sess.source_map().new_source_file(name, source))
}

/// Creates a new parser, handling errors as appropriate if the file doesn't exist.
/// If a span is given, that is used on an error as the source of the problem.
pub fn new_parser_from_file<'a>(sess: &'a ParseSess, path: &Path, sp: Option<Span>) -> Parser<'a> {
    source_file_to_parser(sess, file_to_source_file(sess, path, sp))
}

/// Given a `source_file` and config, returns a parser.
fn source_file_to_parser(sess: &ParseSess, source_file: Lrc<SourceFile>) -> Parser<'_> {
    panictry_buffer!(&sess.span_diagnostic, maybe_source_file_to_parser(sess, source_file))
}

/// Given a `source_file` and config, return a parser. Returns any buffered errors from lexing the
/// initial token stream.
fn maybe_source_file_to_parser(
    sess: &ParseSess,
    source_file: Lrc<SourceFile>,
) -> Result<Parser<'_>, Vec<Diagnostic>> {
    let end_pos = source_file.end_pos;
    let (stream, unclosed_delims) = maybe_file_to_stream(sess, source_file, None)?;
    let mut parser = stream_to_parser(sess, stream, None);
    parser.unclosed_delims = unclosed_delims;
    if parser.token == token::Eof {
        parser.token.span = Span::new(end_pos, end_pos, parser.token.span.ctxt(), None);
    }

    Ok(parser)
}

// Base abstractions

/// Given a session and a path and an optional span (for error reporting),
/// add the path to the session's source_map and return the new source_file or
/// error when a file can't be read.
fn try_file_to_source_file(
    sess: &ParseSess,
    path: &Path,
    spanopt: Option<Span>,
) -> Result<Lrc<SourceFile>, Diagnostic> {
    sess.source_map().load_file(path).map_err(|e| {
        let msg = format!("couldn't read {}: {}", path.display(), e);
        let mut diag = Diagnostic::new(Level::Fatal, &msg);
        if let Some(sp) = spanopt {
            diag.set_span(sp);
        }
        diag
    })
}

/// Given a session and a path and an optional span (for error reporting),
/// adds the path to the session's `source_map` and returns the new `source_file`.
fn file_to_source_file(sess: &ParseSess, path: &Path, spanopt: Option<Span>) -> Lrc<SourceFile> {
    match try_file_to_source_file(sess, path, spanopt) {
        Ok(source_file) => source_file,
        Err(d) => {
            sess.span_diagnostic.emit_diagnostic(&d);
            FatalError.raise();
        }
    }
}

/// Given a `source_file`, produces a sequence of token trees.
pub fn source_file_to_stream(
    sess: &ParseSess,
    source_file: Lrc<SourceFile>,
    override_span: Option<Span>,
) -> (TokenStream, Vec<lexer::UnmatchedBrace>) {
    panictry_buffer!(&sess.span_diagnostic, maybe_file_to_stream(sess, source_file, override_span))
}

/// Given a source file, produces a sequence of token trees. Returns any buffered errors from
/// parsing the token stream.
pub fn maybe_file_to_stream(
    sess: &ParseSess,
    source_file: Lrc<SourceFile>,
    override_span: Option<Span>,
) -> Result<(TokenStream, Vec<lexer::UnmatchedBrace>), Vec<Diagnostic>> {
    let src = source_file.src.as_ref().unwrap_or_else(|| {
        sess.span_diagnostic.bug(&format!(
            "cannot lex `source_file` without source: {}",
            sess.source_map().filename_for_diagnostics(&source_file.name)
        ));
    });

    let (token_trees, unmatched_braces) =
        lexer::parse_token_trees(sess, src.as_str(), source_file.start_pos, override_span);

    match token_trees {
        Ok(stream) => Ok((stream, unmatched_braces)),
        Err(err) => {
            let mut buffer = Vec::with_capacity(1);
            err.buffer(&mut buffer);
            // Not using `emit_unclosed_delims` to use `db.buffer`
            for unmatched in unmatched_braces {
                if let Some(err) = make_unclosed_delims_error(unmatched, &sess) {
                    err.buffer(&mut buffer);
                }
            }
            Err(buffer)
        }
    }
}

/// Given a stream and the `ParseSess`, produces a parser.
pub fn stream_to_parser<'a>(
    sess: &'a ParseSess,
    stream: TokenStream,
    subparser_name: Option<&'static str>,
) -> Parser<'a> {
    Parser::new(sess, stream, false, subparser_name)
}

/// Runs the given subparser `f` on the tokens of the given `attr`'s item.
pub fn parse_in<'a, T>(
    sess: &'a ParseSess,
    tts: TokenStream,
    name: &'static str,
    mut f: impl FnMut(&mut Parser<'a>) -> PResult<'a, T>,
) -> PResult<'a, T> {
    let mut parser = Parser::new(sess, tts, false, Some(name));
    let result = f(&mut parser)?;
    if parser.token != token::Eof {
        parser.unexpected()?;
    }
    Ok(result)
}

// NOTE(Centril): The following probably shouldn't be here but it acknowledges the
// fact that architecturally, we are using parsing (read on below to understand why).

pub fn nt_to_tokenstream(
    nt: &Nonterminal,
    sess: &ParseSess,
    synthesize_tokens: CanSynthesizeMissingTokens,
) -> TokenStream {
    // A `Nonterminal` is often a parsed AST item. At this point we now
    // need to convert the parsed AST to an actual token stream, e.g.
    // un-parse it basically.
    //
    // Unfortunately there's not really a great way to do that in a
    // guaranteed lossless fashion right now. The fallback here is to just
    // stringify the AST node and reparse it, but this loses all span
    // information.
    //
    // As a result, some AST nodes are annotated with the token stream they
    // came from. Here we attempt to extract these lossless token streams
    // before we fall back to the stringification.

    let convert_tokens =
        |tokens: Option<&LazyTokenStream>| Some(tokens?.create_token_stream().to_tokenstream());

    let tokens = match *nt {
        Nonterminal::NtItem(ref item) => prepend_attrs(&item.attrs, item.tokens.as_ref()),
        Nonterminal::NtBlock(ref block) => convert_tokens(block.tokens.as_ref()),
        Nonterminal::NtStmt(ref stmt) if let ast::StmtKind::Empty = stmt.kind => {
            let tokens = AttrAnnotatedTokenStream::new(vec![(
                tokenstream::AttrAnnotatedTokenTree::Token(Token::new(
                    TokenKind::Semi,
                    stmt.span,
                )),
                Spacing::Alone,
            )]);
            prepend_attrs(&stmt.attrs(), Some(&LazyTokenStream::new(tokens)))
        }
        Nonterminal::NtStmt(ref stmt) => prepend_attrs(&stmt.attrs(), stmt.tokens()),
        Nonterminal::NtPat(ref pat) => convert_tokens(pat.tokens.as_ref()),
        Nonterminal::NtTy(ref ty) => convert_tokens(ty.tokens.as_ref()),
        Nonterminal::NtIdent(ident, is_raw) => {
            Some(tokenstream::TokenTree::token(token::Ident(ident.name, is_raw), ident.span).into())
        }
        Nonterminal::NtLifetime(ident) => {
            Some(tokenstream::TokenTree::token(token::Lifetime(ident.name), ident.span).into())
        }
        Nonterminal::NtMeta(ref attr) => convert_tokens(attr.tokens.as_ref()),
        Nonterminal::NtPath(ref path) => convert_tokens(path.tokens.as_ref()),
        Nonterminal::NtVis(ref vis) => convert_tokens(vis.tokens.as_ref()),
        Nonterminal::NtTT(ref tt) => Some(tt.clone().into()),
        Nonterminal::NtExpr(ref expr) | Nonterminal::NtLiteral(ref expr) => {
            prepend_attrs(&expr.attrs, expr.tokens.as_ref())
        }
    };

    if let Some(tokens) = tokens {
        return tokens;
    } else if matches!(synthesize_tokens, CanSynthesizeMissingTokens::Yes) {
        return fake_token_stream(sess, nt);
    } else {
        panic!(
            "Missing tokens for nt {:?} at {:?}: {:?}",
            nt,
            nt.span(),
            pprust::nonterminal_to_string(nt)
        );
    }
}

fn prepend_attrs(attrs: &[Attribute], tokens: Option<&LazyTokenStream>) -> Option<TokenStream> {
    let tokens = tokens?;
    if attrs.is_empty() {
        return Some(tokens.create_token_stream().to_tokenstream());
    }
    let attr_data = AttributesData { attrs: attrs.to_vec().into(), tokens: tokens.clone() };
    let wrapped = AttrAnnotatedTokenStream::new(vec![(
        AttrAnnotatedTokenTree::Attributes(attr_data),
        Spacing::Alone,
    )]);
    Some(wrapped.to_tokenstream())
}

pub fn fake_token_stream(sess: &ParseSess, nt: &Nonterminal) -> TokenStream {
    let source = pprust::nonterminal_to_string(nt);
    let filename = FileName::macro_expansion_source_code(&source);
    parse_stream_from_source_str(filename, source, sess, Some(nt.span()))
}

pub fn parse_cfg_attr(
    attr: &Attribute,
    parse_sess: &ParseSess,
) -> Option<(MetaItem, Vec<(AttrItem, Span)>)> {
    match attr.get_normal_item().args {
        ast::MacArgs::Delimited(dspan, delim, ref tts) if !tts.is_empty() => {
            let msg = "wrong `cfg_attr` delimiters";
            crate::validate_attr::check_meta_bad_delim(parse_sess, dspan, delim, msg);
            match parse_in(parse_sess, tts.clone(), "`cfg_attr` input", |p| p.parse_cfg_attr()) {
                Ok(r) => return Some(r),
                Err(mut e) => {
                    e.help(&format!("the valid syntax is `{}`", CFG_ATTR_GRAMMAR_HELP))
                        .note(CFG_ATTR_NOTE_REF)
                        .emit();
                }
            }
        }
        _ => error_malformed_cfg_attr_missing(attr.span, parse_sess),
    }
    None
}

const CFG_ATTR_GRAMMAR_HELP: &str = "#[cfg_attr(condition, attribute, other_attribute, ...)]";
const CFG_ATTR_NOTE_REF: &str = "for more information, visit \
    <https://doc.rust-lang.org/reference/conditional-compilation.html\
    #the-cfg_attr-attribute>";

fn error_malformed_cfg_attr_missing(span: Span, parse_sess: &ParseSess) {
    parse_sess
        .span_diagnostic
        .struct_span_err(span, "malformed `cfg_attr` attribute input")
        .span_suggestion(
            span,
            "missing condition and attribute",
            CFG_ATTR_GRAMMAR_HELP.to_string(),
            Applicability::HasPlaceholders,
        )
        .note(CFG_ATTR_NOTE_REF)
        .emit();
}
