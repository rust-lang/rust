//! The main parser interface.

#![feature(bool_to_option)]
#![feature(crate_visibility_modifier)]
#![feature(bindings_after_at)]
#![feature(try_blocks)]
#![feature(or_patterns)]

use rustc_ast::ast;
use rustc_ast::token::{self, Nonterminal};
use rustc_ast::tokenstream::{self, TokenStream, TokenTree};
use rustc_ast_pretty::pprust;
use rustc_data_structures::sync::Lrc;
use rustc_errors::{Diagnostic, FatalError, Level, PResult};
use rustc_session::parse::ParseSess;
use rustc_span::{FileName, SourceFile, Span};

use std::path::Path;
use std::str;

use log::{debug, info};

pub const MACRO_ARGUMENTS: Option<&'static str> = Some("macro arguments");

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
/// If a span is given, that is used on an error as the as the source of the problem.
pub fn new_parser_from_file<'a>(sess: &'a ParseSess, path: &Path, sp: Option<Span>) -> Parser<'a> {
    source_file_to_parser(sess, file_to_source_file(sess, path, sp))
}

/// Creates a new parser, returning buffered diagnostics if the file doesn't exist,
/// or from lexing the initial token stream.
pub fn maybe_new_parser_from_file<'a>(
    sess: &'a ParseSess,
    path: &Path,
) -> Result<Parser<'a>, Vec<Diagnostic>> {
    let file = try_file_to_source_file(sess, path, None).map_err(|db| vec![db])?;
    maybe_source_file_to_parser(sess, file)
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
        parser.token.span = Span::new(end_pos, end_pos, parser.token.span.ctxt());
    }

    Ok(parser)
}

// Must preserve old name for now, because `quote!` from the *existing*
// compiler expands into it.
pub fn new_parser_from_tts(sess: &ParseSess, tts: Vec<TokenTree>) -> Parser<'_> {
    stream_to_parser(sess, tts.into_iter().collect(), crate::MACRO_ARGUMENTS)
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
    let srdr = lexer::StringReader::new(sess, source_file, override_span);
    let (token_trees, unmatched_braces) = srdr.into_token_trees();

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

pub fn nt_to_tokenstream(nt: &Nonterminal, sess: &ParseSess, span: Span) -> TokenStream {
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
    let tokens = match *nt {
        Nonterminal::NtItem(ref item) => {
            prepend_attrs(sess, &item.attrs, item.tokens.as_ref(), span)
        }
        Nonterminal::NtIdent(ident, is_raw) => {
            Some(tokenstream::TokenTree::token(token::Ident(ident.name, is_raw), ident.span).into())
        }
        Nonterminal::NtLifetime(ident) => {
            Some(tokenstream::TokenTree::token(token::Lifetime(ident.name), ident.span).into())
        }
        Nonterminal::NtTT(ref tt) => Some(tt.clone().into()),
        Nonterminal::NtExpr(ref expr) => {
            if expr.tokens.is_none() {
                debug!("missing tokens for expr {:?}", expr);
            }
            prepend_attrs(sess, &expr.attrs, expr.tokens.as_ref(), span)
        }
        _ => None,
    };

    // FIXME(#43081): Avoid this pretty-print + reparse hack
    let source = pprust::nonterminal_to_string(nt);
    let filename = FileName::macro_expansion_source_code(&source);
    let tokens_for_real = parse_stream_from_source_str(filename, source, sess, Some(span));

    // During early phases of the compiler the AST could get modified
    // directly (e.g., attributes added or removed) and the internal cache
    // of tokens my not be invalidated or updated. Consequently if the
    // "lossless" token stream disagrees with our actual stringification
    // (which has historically been much more battle-tested) then we go
    // with the lossy stream anyway (losing span information).
    //
    // Note that the comparison isn't `==` here to avoid comparing spans,
    // but it *also* is a "probable" equality which is a pretty weird
    // definition. We mostly want to catch actual changes to the AST
    // like a `#[cfg]` being processed or some weird `macro_rules!`
    // expansion.
    //
    // What we *don't* want to catch is the fact that a user-defined
    // literal like `0xf` is stringified as `15`, causing the cached token
    // stream to not be literal `==` token-wise (ignoring spans) to the
    // token stream we got from stringification.
    //
    // Instead the "probably equal" check here is "does each token
    // recursively have the same discriminant?" We basically don't look at
    // the token values here and assume that such fine grained token stream
    // modifications, including adding/removing typically non-semantic
    // tokens such as extra braces and commas, don't happen.
    if let Some(tokens) = tokens {
        if tokens.probably_equal_for_proc_macro(&tokens_for_real) {
            return tokens;
        }
        info!(
            "cached tokens found, but they're not \"probably equal\", \
                going with stringified version"
        );
        info!("cached tokens: {:?}", tokens);
        info!("reparsed tokens: {:?}", tokens_for_real);
    }
    tokens_for_real
}

fn prepend_attrs(
    sess: &ParseSess,
    attrs: &[ast::Attribute],
    tokens: Option<&tokenstream::TokenStream>,
    span: rustc_span::Span,
) -> Option<tokenstream::TokenStream> {
    let tokens = tokens?;
    if attrs.is_empty() {
        return Some(tokens.clone());
    }
    let mut builder = tokenstream::TokenStreamBuilder::new();
    for attr in attrs {
        assert_eq!(
            attr.style,
            ast::AttrStyle::Outer,
            "inner attributes should prevent cached tokens from existing"
        );

        let source = pprust::attribute_to_string(attr);
        let macro_filename = FileName::macro_expansion_source_code(&source);

        let item = match attr.kind {
            ast::AttrKind::Normal(ref item) => item,
            ast::AttrKind::DocComment(_) => {
                let stream = parse_stream_from_source_str(macro_filename, source, sess, Some(span));
                builder.push(stream);
                continue;
            }
        };

        // synthesize # [ $path $tokens ] manually here
        let mut brackets = tokenstream::TokenStreamBuilder::new();

        // For simple paths, push the identifier directly
        if item.path.segments.len() == 1 && item.path.segments[0].args.is_none() {
            let ident = item.path.segments[0].ident;
            let token = token::Ident(ident.name, ident.as_str().starts_with("r#"));
            brackets.push(tokenstream::TokenTree::token(token, ident.span));

        // ... and for more complicated paths, fall back to a reparse hack that
        // should eventually be removed.
        } else {
            let stream = parse_stream_from_source_str(macro_filename, source, sess, Some(span));
            brackets.push(stream);
        }

        brackets.push(item.args.outer_tokens());

        // The span we list here for `#` and for `[ ... ]` are both wrong in
        // that it encompasses more than each token, but it hopefully is "good
        // enough" for now at least.
        builder.push(tokenstream::TokenTree::token(token::Pound, attr.span));
        let delim_span = tokenstream::DelimSpan::from_single(attr.span);
        builder.push(tokenstream::TokenTree::Delimited(
            delim_span,
            token::DelimToken::Bracket,
            brackets.build(),
        ));
    }
    builder.push(tokens.clone());
    Some(builder.build())
}
