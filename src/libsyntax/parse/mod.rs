//! The main parser interface.

use crate::ast::{self, CrateConfig, NodeId};
use crate::early_buffered_lints::{BufferedEarlyLint, BufferedEarlyLintId};
use crate::source_map::{SourceMap, FilePathMapping};
use crate::feature_gate::UnstableFeatures;
use crate::parse::parser::Parser;
use crate::parse::parser::emit_unclosed_delims;
use crate::parse::token::TokenKind;
use crate::tokenstream::{TokenStream, TokenTree};
use crate::diagnostics::plugin::ErrorMap;
use crate::print::pprust;
use crate::symbol::Symbol;

use errors::{Applicability, FatalError, Level, Handler, ColorConfig, Diagnostic, DiagnosticBuilder};
use rustc_data_structures::sync::{Lrc, Lock, Once};
use syntax_pos::{Span, SourceFile, FileName, MultiSpan};
use syntax_pos::edition::Edition;

use rustc_data_structures::fx::{FxHashSet, FxHashMap};
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::str;

#[cfg(test)]
mod tests;

#[macro_use]
pub mod parser;
pub mod attr;
pub mod lexer;
pub mod token;

crate mod classify;
crate mod diagnostics;
crate mod literal;
crate mod unescape_error_reporting;

pub type PResult<'a, T> = Result<T, DiagnosticBuilder<'a>>;

/// Info about a parsing session.
pub struct ParseSess {
    pub span_diagnostic: Handler,
    pub unstable_features: UnstableFeatures,
    pub config: CrateConfig,
    pub edition: Edition,
    pub missing_fragment_specifiers: Lock<FxHashSet<Span>>,
    /// Places where raw identifiers were used. This is used for feature-gating raw identifiers.
    pub raw_identifier_spans: Lock<Vec<Span>>,
    /// The registered diagnostics codes.
    crate registered_diagnostics: Lock<ErrorMap>,
    /// Used to determine and report recursive module inclusions.
    included_mod_stack: Lock<Vec<PathBuf>>,
    source_map: Lrc<SourceMap>,
    pub buffered_lints: Lock<Vec<BufferedEarlyLint>>,
    /// Contains the spans of block expressions that could have been incomplete based on the
    /// operation token that followed it, but that the parser cannot identify without further
    /// analysis.
    pub ambiguous_block_expr_parse: Lock<FxHashMap<Span, Span>>,
    pub param_attr_spans: Lock<Vec<Span>>,
    // Places where `let` exprs were used and should be feature gated according to `let_chains`.
    pub let_chains_spans: Lock<Vec<Span>>,
    // Places where `async || ..` exprs were used and should be feature gated.
    pub async_closure_spans: Lock<Vec<Span>>,
    pub injected_crate_name: Once<Symbol>,
}

impl ParseSess {
    pub fn new(file_path_mapping: FilePathMapping) -> Self {
        let cm = Lrc::new(SourceMap::new(file_path_mapping));
        let handler = Handler::with_tty_emitter(ColorConfig::Auto,
                                                true,
                                                None,
                                                Some(cm.clone()));
        ParseSess::with_span_handler(handler, cm)
    }

    pub fn with_span_handler(handler: Handler, source_map: Lrc<SourceMap>) -> ParseSess {
        ParseSess {
            span_diagnostic: handler,
            unstable_features: UnstableFeatures::from_environment(),
            config: FxHashSet::default(),
            missing_fragment_specifiers: Lock::new(FxHashSet::default()),
            raw_identifier_spans: Lock::new(Vec::new()),
            registered_diagnostics: Lock::new(ErrorMap::new()),
            included_mod_stack: Lock::new(vec![]),
            source_map,
            buffered_lints: Lock::new(vec![]),
            edition: Edition::from_session(),
            ambiguous_block_expr_parse: Lock::new(FxHashMap::default()),
            param_attr_spans: Lock::new(Vec::new()),
            let_chains_spans: Lock::new(Vec::new()),
            async_closure_spans: Lock::new(Vec::new()),
            injected_crate_name: Once::new(),
        }
    }

    #[inline]
    pub fn source_map(&self) -> &SourceMap {
        &self.source_map
    }

    pub fn buffer_lint<S: Into<MultiSpan>>(&self,
        lint_id: BufferedEarlyLintId,
        span: S,
        id: NodeId,
        msg: &str,
    ) {
        self.buffered_lints.with_lock(|buffered_lints| {
            buffered_lints.push(BufferedEarlyLint{
                span: span.into(),
                id,
                msg: msg.into(),
                lint_id,
            });
        });
    }

    /// Extend an error with a suggestion to wrap an expression with parentheses to allow the
    /// parser to continue parsing the following operation as part of the same expression.
    pub fn expr_parentheses_needed(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        span: Span,
        alt_snippet: Option<String>,
    ) {
        if let Some(snippet) = self.source_map().span_to_snippet(span).ok().or(alt_snippet) {
            err.span_suggestion(
                span,
                "parentheses are required to parse this as an expression",
                format!("({})", snippet),
                Applicability::MachineApplicable,
            );
        }
    }
}

#[derive(Clone)]
pub struct Directory<'a> {
    pub path: Cow<'a, Path>,
    pub ownership: DirectoryOwnership,
}

#[derive(Copy, Clone)]
pub enum DirectoryOwnership {
    Owned {
        // None if `mod.rs`, `Some("foo")` if we're in `foo.rs`
        relative: Option<ast::Ident>,
    },
    UnownedViaBlock,
    UnownedViaMod(bool /* legacy warnings? */),
}

// a bunch of utility functions of the form parse_<thing>_from_<source>
// where <thing> includes crate, expr, item, stmt, tts, and one that
// uses a HOF to parse anything, and <source> includes file and
// source_str.

pub fn parse_crate_from_file<'a>(input: &Path, sess: &'a ParseSess) -> PResult<'a, ast::Crate> {
    let mut parser = new_parser_from_file(sess, input);
    parser.parse_crate_mod()
}

pub fn parse_crate_attrs_from_file<'a>(input: &Path, sess: &'a ParseSess)
                                       -> PResult<'a, Vec<ast::Attribute>> {
    let mut parser = new_parser_from_file(sess, input);
    parser.parse_inner_attributes()
}

pub fn parse_crate_from_source_str(name: FileName, source: String, sess: &ParseSess)
                                       -> PResult<'_, ast::Crate> {
    new_parser_from_source_str(sess, name, source).parse_crate_mod()
}

pub fn parse_crate_attrs_from_source_str(name: FileName, source: String, sess: &ParseSess)
                                             -> PResult<'_, Vec<ast::Attribute>> {
    new_parser_from_source_str(sess, name, source).parse_inner_attributes()
}

pub fn parse_stream_from_source_str(
    name: FileName,
    source: String,
    sess: &ParseSess,
    override_span: Option<Span>,
) -> TokenStream {
    let (stream, mut errors) = source_file_to_stream(
        sess,
        sess.source_map().new_source_file(name, source),
        override_span,
    );
    emit_unclosed_delims(&mut errors, &sess.span_diagnostic);
    stream
}

/// Creates a new parser from a source string.
pub fn new_parser_from_source_str(sess: &ParseSess, name: FileName, source: String) -> Parser<'_> {
    panictry_buffer!(&sess.span_diagnostic, maybe_new_parser_from_source_str(sess, name, source))
}

/// Creates a new parser from a source string. Returns any buffered errors from lexing the initial
/// token stream.
pub fn maybe_new_parser_from_source_str(sess: &ParseSess, name: FileName, source: String)
    -> Result<Parser<'_>, Vec<Diagnostic>>
{
    let mut parser = maybe_source_file_to_parser(sess,
                                                 sess.source_map().new_source_file(name, source))?;
    parser.recurse_into_file_modules = false;
    Ok(parser)
}

/// Creates a new parser, handling errors as appropriate
/// if the file doesn't exist
pub fn new_parser_from_file<'a>(sess: &'a ParseSess, path: &Path) -> Parser<'a> {
    source_file_to_parser(sess, file_to_source_file(sess, path, None))
}

/// Creates a new parser, returning buffered diagnostics if the file doesn't
/// exist or from lexing the initial token stream.
pub fn maybe_new_parser_from_file<'a>(sess: &'a ParseSess, path: &Path)
    -> Result<Parser<'a>, Vec<Diagnostic>> {
    let file = try_file_to_source_file(sess, path, None).map_err(|db| vec![db])?;
    maybe_source_file_to_parser(sess, file)
}

/// Given a session, a crate config, a path, and a span, add
/// the file at the given path to the source_map, and return a parser.
/// On an error, use the given span as the source of the problem.
pub fn new_sub_parser_from_file<'a>(sess: &'a ParseSess,
                                    path: &Path,
                                    directory_ownership: DirectoryOwnership,
                                    module_name: Option<String>,
                                    sp: Span) -> Parser<'a> {
    let mut p = source_file_to_parser(sess, file_to_source_file(sess, path, Some(sp)));
    p.directory.ownership = directory_ownership;
    p.root_module_name = module_name;
    p
}

/// Given a source_file and config, return a parser
fn source_file_to_parser(sess: &ParseSess, source_file: Lrc<SourceFile>) -> Parser<'_> {
    panictry_buffer!(&sess.span_diagnostic,
                     maybe_source_file_to_parser(sess, source_file))
}

/// Given a source_file and config, return a parser. Returns any buffered errors from lexing the
/// initial token stream.
fn maybe_source_file_to_parser(
    sess: &ParseSess,
    source_file: Lrc<SourceFile>,
) -> Result<Parser<'_>, Vec<Diagnostic>> {
    let end_pos = source_file.end_pos;
    let (stream, unclosed_delims) = maybe_file_to_stream(sess, source_file, None)?;
    let mut parser = stream_to_parser(sess, stream, None);
    parser.unclosed_delims = unclosed_delims;
    if parser.token == token::Eof && parser.token.span.is_dummy() {
        parser.token.span = Span::new(end_pos, end_pos, parser.token.span.ctxt());
    }

    Ok(parser)
}

// must preserve old name for now, because quote! from the *existing*
// compiler expands into it
pub fn new_parser_from_tts(sess: &ParseSess, tts: Vec<TokenTree>) -> Parser<'_> {
    stream_to_parser(sess, tts.into_iter().collect(), crate::MACRO_ARGUMENTS)
}


// base abstractions

/// Given a session and a path and an optional span (for error reporting),
/// add the path to the session's source_map and return the new source_file or
/// error when a file can't be read.
fn try_file_to_source_file(sess: &ParseSess, path: &Path, spanopt: Option<Span>)
                   -> Result<Lrc<SourceFile>, Diagnostic> {
    sess.source_map().load_file(path)
    .map_err(|e| {
        let msg = format!("couldn't read {}: {}", path.display(), e);
        let mut diag = Diagnostic::new(Level::Fatal, &msg);
        if let Some(sp) = spanopt {
            diag.set_span(sp);
        }
        diag
    })
}

/// Given a session and a path and an optional span (for error reporting),
/// add the path to the session's `source_map` and return the new `source_file`.
fn file_to_source_file(sess: &ParseSess, path: &Path, spanopt: Option<Span>)
                   -> Lrc<SourceFile> {
    match try_file_to_source_file(sess, path, spanopt) {
        Ok(source_file) => source_file,
        Err(d) => {
            DiagnosticBuilder::new_diagnostic(&sess.span_diagnostic, d).emit();
            FatalError.raise();
        }
    }
}

/// Given a source_file, produces a sequence of token trees.
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
                let mut db = sess.span_diagnostic.struct_span_err(unmatched.found_span, &format!(
                    "incorrect close delimiter: `{}`",
                    pprust::token_kind_to_string(&token::CloseDelim(unmatched.found_delim)),
                ));
                db.span_label(unmatched.found_span, "incorrect close delimiter");
                if let Some(sp) = unmatched.candidate_span {
                    db.span_label(sp, "close delimiter possibly meant for this");
                }
                if let Some(sp) = unmatched.unclosed_span {
                    db.span_label(sp, "un-closed delimiter");
                }
                db.buffer(&mut buffer);
            }
            Err(buffer)
        }
    }
}

/// Given stream and the `ParseSess`, produces a parser.
pub fn stream_to_parser<'a>(
    sess: &'a ParseSess,
    stream: TokenStream,
    subparser_name: Option<&'static str>,
) -> Parser<'a> {
    Parser::new(sess, stream, None, true, false, subparser_name)
}

/// Given stream, the `ParseSess` and the base directory, produces a parser.
///
/// Use this function when you are creating a parser from the token stream
/// and also care about the current working directory of the parser (e.g.,
/// you are trying to resolve modules defined inside a macro invocation).
///
/// # Note
///
/// The main usage of this function is outside of rustc, for those who uses
/// libsyntax as a library. Please do not remove this function while refactoring
/// just because it is not used in rustc codebase!
pub fn stream_to_parser_with_base_dir<'a>(
    sess: &'a ParseSess,
    stream: TokenStream,
    base_dir: Directory<'a>,
) -> Parser<'a> {
    Parser::new(sess, stream, Some(base_dir), true, false, None)
}

/// A sequence separator.
pub struct SeqSep {
    /// The separator token.
    pub sep: Option<TokenKind>,
    /// `true` if a trailing separator is allowed.
    pub trailing_sep_allowed: bool,
}

impl SeqSep {
    pub fn trailing_allowed(t: TokenKind) -> SeqSep {
        SeqSep {
            sep: Some(t),
            trailing_sep_allowed: true,
        }
    }

    pub fn none() -> SeqSep {
        SeqSep {
            sep: None,
            trailing_sep_allowed: false,
        }
    }
}
