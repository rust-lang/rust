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

use errors::{Applicability, FatalError, Level, Handler, ColorConfig, Diagnostic, DiagnosticBuilder};
use rustc_data_structures::sync::{Lrc, Lock};
use syntax_pos::{Span, SourceFile, FileName, MultiSpan};
use syntax_pos::edition::Edition;

use rustc_data_structures::fx::{FxHashSet, FxHashMap};
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::str;

pub type PResult<'a, T> = Result<T, DiagnosticBuilder<'a>>;

#[macro_use]
pub mod parser;
pub mod attr;
pub mod lexer;
pub mod token;

crate mod classify;
crate mod diagnostics;
crate mod literal;
crate mod unescape;
crate mod unescape_error_reporting;

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
    let srdr = lexer::StringReader::new_or_buffered_errs(sess, source_file, override_span)?;
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
    /// The seperator token.
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{self, Name, PatKind};
    use crate::attr::first_attr_value_str_by_name;
    use crate::ptr::P;
    use crate::parse::token::Token;
    use crate::print::pprust::item_to_string;
    use crate::symbol::{kw, sym};
    use crate::tokenstream::{DelimSpan, TokenTree};
    use crate::util::parser_testing::string_to_stream;
    use crate::util::parser_testing::{string_to_expr, string_to_item};
    use crate::with_default_globals;
    use syntax_pos::{Span, BytePos, Pos, NO_EXPANSION};

    /// Parses an item.
    ///
    /// Returns `Ok(Some(item))` when successful, `Ok(None)` when no item was found, and `Err`
    /// when a syntax error occurred.
    fn parse_item_from_source_str(name: FileName, source: String, sess: &ParseSess)
                                        -> PResult<'_, Option<P<ast::Item>>> {
        new_parser_from_source_str(sess, name, source).parse_item()
    }

    // produce a syntax_pos::span
    fn sp(a: u32, b: u32) -> Span {
        Span::new(BytePos(a), BytePos(b), NO_EXPANSION)
    }

    #[should_panic]
    #[test] fn bad_path_expr_1() {
        with_default_globals(|| {
            string_to_expr("::abc::def::return".to_string());
        })
    }

    // check the token-tree-ization of macros
    #[test]
    fn string_to_tts_macro () {
        with_default_globals(|| {
            let tts: Vec<_> =
                string_to_stream("macro_rules! zip (($a)=>($a))".to_string()).trees().collect();
            let tts: &[TokenTree] = &tts[..];

            match tts {
                [
                    TokenTree::Token(Token { kind: token::Ident(name_macro_rules, false), .. }),
                    TokenTree::Token(Token { kind: token::Not, .. }),
                    TokenTree::Token(Token { kind: token::Ident(name_zip, false), .. }),
                    TokenTree::Delimited(_, macro_delim,  macro_tts)
                ]
                if name_macro_rules == &sym::macro_rules && name_zip.as_str() == "zip" => {
                    let tts = &macro_tts.trees().collect::<Vec<_>>();
                    match &tts[..] {
                        [
                            TokenTree::Delimited(_, first_delim, first_tts),
                            TokenTree::Token(Token { kind: token::FatArrow, .. }),
                            TokenTree::Delimited(_, second_delim, second_tts),
                        ]
                        if macro_delim == &token::Paren => {
                            let tts = &first_tts.trees().collect::<Vec<_>>();
                            match &tts[..] {
                                [
                                    TokenTree::Token(Token { kind: token::Dollar, .. }),
                                    TokenTree::Token(Token { kind: token::Ident(name, false), .. }),
                                ]
                                if first_delim == &token::Paren && name.as_str() == "a" => {},
                                _ => panic!("value 3: {:?} {:?}", first_delim, first_tts),
                            }
                            let tts = &second_tts.trees().collect::<Vec<_>>();
                            match &tts[..] {
                                [
                                    TokenTree::Token(Token { kind: token::Dollar, .. }),
                                    TokenTree::Token(Token { kind: token::Ident(name, false), .. }),
                                ]
                                if second_delim == &token::Paren && name.as_str() == "a" => {},
                                _ => panic!("value 4: {:?} {:?}", second_delim, second_tts),
                            }
                        },
                        _ => panic!("value 2: {:?} {:?}", macro_delim, macro_tts),
                    }
                },
                _ => panic!("value: {:?}",tts),
            }
        })
    }

    #[test]
    fn string_to_tts_1() {
        with_default_globals(|| {
            let tts = string_to_stream("fn a (b : i32) { b; }".to_string());

            let expected = TokenStream::new(vec![
                TokenTree::token(token::Ident(kw::Fn, false), sp(0, 2)).into(),
                TokenTree::token(token::Ident(Name::intern("a"), false), sp(3, 4)).into(),
                TokenTree::Delimited(
                    DelimSpan::from_pair(sp(5, 6), sp(13, 14)),
                    token::DelimToken::Paren,
                    TokenStream::new(vec![
                        TokenTree::token(token::Ident(Name::intern("b"), false), sp(6, 7)).into(),
                        TokenTree::token(token::Colon, sp(8, 9)).into(),
                        TokenTree::token(token::Ident(sym::i32, false), sp(10, 13)).into(),
                    ]).into(),
                ).into(),
                TokenTree::Delimited(
                    DelimSpan::from_pair(sp(15, 16), sp(20, 21)),
                    token::DelimToken::Brace,
                    TokenStream::new(vec![
                        TokenTree::token(token::Ident(Name::intern("b"), false), sp(17, 18)).into(),
                        TokenTree::token(token::Semi, sp(18, 19)).into(),
                    ]).into(),
                ).into()
            ]);

            assert_eq!(tts, expected);
        })
    }

    #[test] fn parse_use() {
        with_default_globals(|| {
            let use_s = "use foo::bar::baz;";
            let vitem = string_to_item(use_s.to_string()).unwrap();
            let vitem_s = item_to_string(&vitem);
            assert_eq!(&vitem_s[..], use_s);

            let use_s = "use foo::bar as baz;";
            let vitem = string_to_item(use_s.to_string()).unwrap();
            let vitem_s = item_to_string(&vitem);
            assert_eq!(&vitem_s[..], use_s);
        })
    }

    #[test] fn parse_extern_crate() {
        with_default_globals(|| {
            let ex_s = "extern crate foo;";
            let vitem = string_to_item(ex_s.to_string()).unwrap();
            let vitem_s = item_to_string(&vitem);
            assert_eq!(&vitem_s[..], ex_s);

            let ex_s = "extern crate foo as bar;";
            let vitem = string_to_item(ex_s.to_string()).unwrap();
            let vitem_s = item_to_string(&vitem);
            assert_eq!(&vitem_s[..], ex_s);
        })
    }

    fn get_spans_of_pat_idents(src: &str) -> Vec<Span> {
        let item = string_to_item(src.to_string()).unwrap();

        struct PatIdentVisitor {
            spans: Vec<Span>
        }
        impl<'a> crate::visit::Visitor<'a> for PatIdentVisitor {
            fn visit_pat(&mut self, p: &'a ast::Pat) {
                match p.node {
                    PatKind::Ident(_ , ref spannedident, _) => {
                        self.spans.push(spannedident.span.clone());
                    }
                    _ => {
                        crate::visit::walk_pat(self, p);
                    }
                }
            }
        }
        let mut v = PatIdentVisitor { spans: Vec::new() };
        crate::visit::walk_item(&mut v, &item);
        return v.spans;
    }

    #[test] fn span_of_self_arg_pat_idents_are_correct() {
        with_default_globals(|| {

            let srcs = ["impl z { fn a (&self, &myarg: i32) {} }",
                        "impl z { fn a (&mut self, &myarg: i32) {} }",
                        "impl z { fn a (&'a self, &myarg: i32) {} }",
                        "impl z { fn a (self, &myarg: i32) {} }",
                        "impl z { fn a (self: Foo, &myarg: i32) {} }",
                        ];

            for &src in &srcs {
                let spans = get_spans_of_pat_idents(src);
                let (lo, hi) = (spans[0].lo(), spans[0].hi());
                assert!("self" == &src[lo.to_usize()..hi.to_usize()],
                        "\"{}\" != \"self\". src=\"{}\"",
                        &src[lo.to_usize()..hi.to_usize()], src)
            }
        })
    }

    #[test] fn parse_exprs () {
        with_default_globals(|| {
            // just make sure that they parse....
            string_to_expr("3 + 4".to_string());
            string_to_expr("a::z.froob(b,&(987+3))".to_string());
        })
    }

    #[test] fn attrs_fix_bug () {
        with_default_globals(|| {
            string_to_item("pub fn mk_file_writer(path: &Path, flags: &[FileFlag])
                   -> Result<Box<Writer>, String> {
    #[cfg(windows)]
    fn wb() -> c_int {
      (O_WRONLY | libc::consts::os::extra::O_BINARY) as c_int
    }

    #[cfg(unix)]
    fn wb() -> c_int { O_WRONLY as c_int }

    let mut fflags: c_int = wb();
}".to_string());
        })
    }

    #[test] fn crlf_doc_comments() {
        with_default_globals(|| {
            let sess = ParseSess::new(FilePathMapping::empty());

            let name_1 = FileName::Custom("crlf_source_1".to_string());
            let source = "/// doc comment\r\nfn foo() {}".to_string();
            let item = parse_item_from_source_str(name_1, source, &sess)
                .unwrap().unwrap();
            let doc = first_attr_value_str_by_name(&item.attrs, sym::doc).unwrap();
            assert_eq!(doc.as_str(), "/// doc comment");

            let name_2 = FileName::Custom("crlf_source_2".to_string());
            let source = "/// doc comment\r\n/// line 2\r\nfn foo() {}".to_string();
            let item = parse_item_from_source_str(name_2, source, &sess)
                .unwrap().unwrap();
            let docs = item.attrs.iter().filter(|a| a.path == sym::doc)
                        .map(|a| a.value_str().unwrap().to_string()).collect::<Vec<_>>();
            let b: &[_] = &["/// doc comment".to_string(), "/// line 2".to_string()];
            assert_eq!(&docs[..], b);

            let name_3 = FileName::Custom("clrf_source_3".to_string());
            let source = "/** doc comment\r\n *  with CRLF */\r\nfn foo() {}".to_string();
            let item = parse_item_from_source_str(name_3, source, &sess).unwrap().unwrap();
            let doc = first_attr_value_str_by_name(&item.attrs, sym::doc).unwrap();
            assert_eq!(doc.as_str(), "/** doc comment\n *  with CRLF */");
        });
    }

    #[test]
    fn ttdelim_span() {
        fn parse_expr_from_source_str(
            name: FileName, source: String, sess: &ParseSess
        ) -> PResult<'_, P<ast::Expr>> {
            new_parser_from_source_str(sess, name, source).parse_expr()
        }

        with_default_globals(|| {
            let sess = ParseSess::new(FilePathMapping::empty());
            let expr = parse_expr_from_source_str(PathBuf::from("foo").into(),
                "foo!( fn main() { body } )".to_string(), &sess).unwrap();

            let tts: Vec<_> = match expr.node {
                ast::ExprKind::Mac(ref mac) => mac.node.stream().trees().collect(),
                _ => panic!("not a macro"),
            };

            let span = tts.iter().rev().next().unwrap().span();

            match sess.source_map().span_to_snippet(span) {
                Ok(s) => assert_eq!(&s[..], "{ body }"),
                Err(_) => panic!("could not get snippet"),
            }
        });
    }

    // This tests that when parsing a string (rather than a file) we don't try
    // and read in a file for a module declaration and just parse a stub.
    // See `recurse_into_file_modules` in the parser.
    #[test]
    fn out_of_line_mod() {
        with_default_globals(|| {
            let sess = ParseSess::new(FilePathMapping::empty());
            let item = parse_item_from_source_str(
                PathBuf::from("foo").into(),
                "mod foo { struct S; mod this_does_not_exist; }".to_owned(),
                &sess,
            ).unwrap().unwrap();

            if let ast::ItemKind::Mod(ref m) = item.node {
                assert!(m.items.len() == 2);
            } else {
                panic!();
            }
        });
    }
}
