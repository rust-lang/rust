//! The main parser interface.

#![feature(crate_visibility_modifier)]
#![feature(bindings_after_at)]
#![feature(iter_order_by)]
#![feature(or_patterns)]

use rustc_ast as ast;
use rustc_ast::token::{self, DelimToken, Nonterminal, Token, TokenKind};
use rustc_ast::tokenstream::{self, LazyTokenStream, TokenStream, TokenTree};
use rustc_ast_pretty::pprust;
use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::sync::Lrc;
use rustc_errors::{Diagnostic, FatalError, Level, PResult};
use rustc_session::parse::ParseSess;
use rustc_span::{symbol::kw, FileName, SourceFile, Span, DUMMY_SP};

use smallvec::SmallVec;
use std::cell::RefCell;
use std::mem;
use std::path::Path;
use std::str;

use tracing::{debug, info};

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
        parser.token.span = Span::new(end_pos, end_pos, parser.token.span.ctxt());
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
        sess.span_diagnostic
            .bug(&format!("cannot lex `source_file` without source: {}", source_file.name));
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

    let convert_tokens =
        |tokens: &Option<LazyTokenStream>| tokens.as_ref().map(|t| t.create_token_stream());

    let tokens = match *nt {
        Nonterminal::NtItem(ref item) => prepend_attrs(&item.attrs, item.tokens.as_ref()),
        Nonterminal::NtBlock(ref block) => convert_tokens(&block.tokens),
        Nonterminal::NtStmt(ref stmt) => {
            // FIXME: We currently only collect tokens for `:stmt`
            // matchers in `macro_rules!` macros. When we start collecting
            // tokens for attributes on statements, we will need to prepend
            // attributes here
            convert_tokens(&stmt.tokens)
        }
        Nonterminal::NtPat(ref pat) => convert_tokens(&pat.tokens),
        Nonterminal::NtTy(ref ty) => convert_tokens(&ty.tokens),
        Nonterminal::NtIdent(ident, is_raw) => {
            Some(tokenstream::TokenTree::token(token::Ident(ident.name, is_raw), ident.span).into())
        }
        Nonterminal::NtLifetime(ident) => {
            Some(tokenstream::TokenTree::token(token::Lifetime(ident.name), ident.span).into())
        }
        Nonterminal::NtMeta(ref attr) => convert_tokens(&attr.tokens),
        Nonterminal::NtPath(ref path) => convert_tokens(&path.tokens),
        Nonterminal::NtVis(ref vis) => convert_tokens(&vis.tokens),
        Nonterminal::NtTT(ref tt) => Some(tt.clone().into()),
        Nonterminal::NtExpr(ref expr) | Nonterminal::NtLiteral(ref expr) => {
            if expr.tokens.is_none() {
                debug!("missing tokens for expr {:?}", expr);
            }
            prepend_attrs(&expr.attrs, expr.tokens.as_ref())
        }
    };

    // Caches the stringification of 'good' `TokenStreams` which passed
    // `tokenstream_probably_equal_for_proc_macro`. This allows us to avoid
    // repeatedly stringifying and comparing the same `TokenStream` for deeply
    // nested nonterminals.
    //
    // We cache by the strinification instead of the `TokenStream` to avoid
    // needing to implement `Hash` for `TokenStream`. Note that it's possible to
    // have two distinct `TokenStream`s that stringify to the same result
    // (e.g. if they differ only in hygiene information). However, any
    // information lost during the stringification process is also intentionally
    // ignored by `tokenstream_probably_equal_for_proc_macro`, so it's fine
    // that a single cache entry may 'map' to multiple distinct `TokenStream`s.
    //
    // This is a temporary hack to prevent compilation blowup on certain inputs.
    // The entire pretty-print/retokenize process will be removed soon.
    thread_local! {
        static GOOD_TOKEN_CACHE: RefCell<FxHashSet<String>> = Default::default();
    }

    // FIXME(#43081): Avoid this pretty-print + reparse hack
    // Pretty-print the AST struct without inserting any parenthesis
    // beyond those explicitly written by the user (e.g. `ExpnKind::Paren`).
    // The resulting stream may have incorrect precedence, but it's only
    // ever used for a comparison against the capture tokenstream.
    let source = pprust::nonterminal_to_string_no_extra_parens(nt);
    let filename = FileName::macro_expansion_source_code(&source);
    let reparsed_tokens = parse_stream_from_source_str(filename, source.clone(), sess, Some(span));

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
        if GOOD_TOKEN_CACHE.with(|cache| cache.borrow().contains(&source)) {
            return tokens;
        }

        // Compare with a non-relaxed delim match to start.
        if tokenstream_probably_equal_for_proc_macro(&tokens, &reparsed_tokens, sess, false) {
            GOOD_TOKEN_CACHE.with(|cache| cache.borrow_mut().insert(source.clone()));
            return tokens;
        }

        // The check failed. This time, we pretty-print the AST struct with parenthesis
        // inserted to preserve precedence. This may cause `None`-delimiters in the captured
        // token stream to match up with inserted parenthesis in the reparsed stream.
        let source_with_parens = pprust::nonterminal_to_string(nt);
        let filename_with_parens = FileName::macro_expansion_source_code(&source_with_parens);

        if GOOD_TOKEN_CACHE.with(|cache| cache.borrow().contains(&source_with_parens)) {
            return tokens;
        }

        let reparsed_tokens_with_parens = parse_stream_from_source_str(
            filename_with_parens,
            source_with_parens,
            sess,
            Some(span),
        );

        // Compare with a relaxed delim match - we want inserted parenthesis in the
        // reparsed stream to match `None`-delimiters in the original stream.
        if tokenstream_probably_equal_for_proc_macro(
            &tokens,
            &reparsed_tokens_with_parens,
            sess,
            true,
        ) {
            GOOD_TOKEN_CACHE.with(|cache| cache.borrow_mut().insert(source.clone()));
            return tokens;
        }

        info!(
            "cached tokens found, but they're not \"probably equal\", \
                going with stringified version"
        );
        info!("cached   tokens: {}", pprust::tts_to_string(&tokens));
        info!("reparsed tokens: {}", pprust::tts_to_string(&reparsed_tokens_with_parens));

        info!("cached   tokens debug: {:?}", tokens);
        info!("reparsed tokens debug: {:?}", reparsed_tokens_with_parens);
    }
    reparsed_tokens
}

// See comments in `Nonterminal::to_tokenstream` for why we care about
// *probably* equal here rather than actual equality
//
// This is otherwise the same as `eq_unspanned`, only recursing with a
// different method.
pub fn tokenstream_probably_equal_for_proc_macro(
    tokens: &TokenStream,
    reparsed_tokens: &TokenStream,
    sess: &ParseSess,
    relaxed_delim_match: bool,
) -> bool {
    // When checking for `probably_eq`, we ignore certain tokens that aren't
    // preserved in the AST. Because they are not preserved, the pretty
    // printer arbitrarily adds or removes them when printing as token
    // streams, making a comparison between a token stream generated from an
    // AST and a token stream which was parsed into an AST more reliable.
    fn semantic_tree(tree: &TokenTree) -> bool {
        if let TokenTree::Token(token) = tree {
            if let
                // The pretty printer tends to add trailing commas to
                // everything, and in particular, after struct fields.
                | token::Comma
                // The pretty printer collapses many semicolons into one.
                | token::Semi
                // We don't preserve leading `|` tokens in patterns, so
                // we ignore them entirely
                | token::BinOp(token::BinOpToken::Or)
                // We don't preserve trailing '+' tokens in trait bounds,
                // so we ignore them entirely
                | token::BinOp(token::BinOpToken::Plus)
                // The pretty printer can turn `$crate` into `::crate_name`
                | token::ModSep = token.kind {
                return false;
            }
        }
        true
    }

    // When comparing two `TokenStream`s, we ignore the `IsJoint` information.
    //
    // However, `rustc_parse::lexer::tokentrees::TokenStreamBuilder` will
    // use `Token.glue` on adjacent tokens with the proper `IsJoint`.
    // Since we are ignoreing `IsJoint`, a 'glued' token (e.g. `BinOp(Shr)`)
    // and its 'split'/'unglued' compoenents (e.g. `Gt, Gt`) are equivalent
    // when determining if two `TokenStream`s are 'probably equal'.
    //
    // Therefore, we use `break_two_token_op` to convert all tokens
    // to the 'unglued' form (if it exists). This ensures that two
    // `TokenStream`s which differ only in how their tokens are glued
    // will be considered 'probably equal', which allows us to keep spans.
    //
    // This is important when the original `TokenStream` contained
    // extra spaces (e.g. `f :: < Vec < _ > > ( ) ;'). These extra spaces
    // will be omitted when we pretty-print, which can cause the original
    // and reparsed `TokenStream`s to differ in the assignment of `IsJoint`,
    // leading to some tokens being 'glued' together in one stream but not
    // the other. See #68489 for more details.
    fn break_tokens(tree: TokenTree) -> impl Iterator<Item = TokenTree> {
        // In almost all cases, we should have either zero or one levels
        // of 'unglueing'. However, in some unusual cases, we may need
        // to iterate breaking tokens mutliple times. For example:
        // '[BinOpEq(Shr)] => [Gt, Ge] -> [Gt, Gt, Eq]'
        let mut token_trees: SmallVec<[_; 2]>;
        if let TokenTree::Token(token) = tree {
            let mut out = SmallVec::<[_; 2]>::new();
            out.push(token);
            // Iterate to fixpoint:
            // * We start off with 'out' containing our initial token, and `temp` empty
            // * If we are able to break any tokens in `out`, then `out` will have
            //   at least one more element than 'temp', so we will try to break tokens
            //   again.
            // * If we cannot break any tokens in 'out', we are done
            loop {
                let mut temp = SmallVec::<[_; 2]>::new();
                let mut changed = false;

                for token in out.into_iter() {
                    if let Some((first, second)) = token.kind.break_two_token_op() {
                        temp.push(Token::new(first, DUMMY_SP));
                        temp.push(Token::new(second, DUMMY_SP));
                        changed = true;
                    } else {
                        temp.push(token);
                    }
                }
                out = temp;
                if !changed {
                    break;
                }
            }
            token_trees = out.into_iter().map(TokenTree::Token).collect();
        } else {
            token_trees = SmallVec::new();
            token_trees.push(tree);
        }
        token_trees.into_iter()
    }

    fn expand_token(tree: TokenTree, sess: &ParseSess) -> impl Iterator<Item = TokenTree> {
        // When checking tokenstreams for 'probable equality', we are comparing
        // a captured (from parsing) `TokenStream` to a reparsed tokenstream.
        // The reparsed Tokenstream will never have `None`-delimited groups,
        // since they are only ever inserted as a result of macro expansion.
        // Therefore, inserting a `None`-delimtied group here (when we
        // convert a nested `Nonterminal` to a tokenstream) would cause
        // a mismatch with the reparsed tokenstream.
        //
        // Note that we currently do not handle the case where the
        // reparsed stream has a `Parenthesis`-delimited group
        // inserted. This will cause a spurious mismatch:
        // issue #75734 tracks resolving this.

        let expanded: SmallVec<[_; 1]> =
            if let TokenTree::Token(Token { kind: TokenKind::Interpolated(nt), span }) = &tree {
                nt_to_tokenstream(nt, sess, *span)
                    .into_trees()
                    .flat_map(|t| expand_token(t, sess))
                    .collect()
            } else {
                // Filter before and after breaking tokens,
                // since we may want to ignore both glued and unglued tokens.
                std::iter::once(tree)
                    .filter(semantic_tree)
                    .flat_map(break_tokens)
                    .filter(semantic_tree)
                    .collect()
            };
        expanded.into_iter()
    }

    // Break tokens after we expand any nonterminals, so that we break tokens
    // that are produced as a result of nonterminal expansion.
    let tokens = tokens.trees().flat_map(|t| expand_token(t, sess));
    let reparsed_tokens = reparsed_tokens.trees().flat_map(|t| expand_token(t, sess));

    tokens.eq_by(reparsed_tokens, |t, rt| {
        tokentree_probably_equal_for_proc_macro(&t, &rt, sess, relaxed_delim_match)
    })
}

// See comments in `Nonterminal::to_tokenstream` for why we care about
// *probably* equal here rather than actual equality
//
// This is otherwise the same as `eq_unspanned`, only recursing with a
// different method.
pub fn tokentree_probably_equal_for_proc_macro(
    token: &TokenTree,
    reparsed_token: &TokenTree,
    sess: &ParseSess,
    relaxed_delim_match: bool,
) -> bool {
    match (token, reparsed_token) {
        (TokenTree::Token(token), TokenTree::Token(reparsed_token)) => {
            token_probably_equal_for_proc_macro(token, reparsed_token)
        }
        (
            TokenTree::Delimited(_, delim, tokens),
            TokenTree::Delimited(_, reparsed_delim, reparsed_tokens),
        ) if delim == reparsed_delim => tokenstream_probably_equal_for_proc_macro(
            tokens,
            reparsed_tokens,
            sess,
            relaxed_delim_match,
        ),
        (TokenTree::Delimited(_, DelimToken::NoDelim, tokens), reparsed_token) => {
            if relaxed_delim_match {
                if let TokenTree::Delimited(_, DelimToken::Paren, reparsed_tokens) = reparsed_token
                {
                    if tokenstream_probably_equal_for_proc_macro(
                        tokens,
                        reparsed_tokens,
                        sess,
                        relaxed_delim_match,
                    ) {
                        return true;
                    }
                }
            }
            tokens.len() == 1
                && tokentree_probably_equal_for_proc_macro(
                    &tokens.trees().next().unwrap(),
                    reparsed_token,
                    sess,
                    relaxed_delim_match,
                )
        }
        _ => false,
    }
}

// See comments in `Nonterminal::to_tokenstream` for why we care about
// *probably* equal here rather than actual equality
fn token_probably_equal_for_proc_macro(first: &Token, other: &Token) -> bool {
    if mem::discriminant(&first.kind) != mem::discriminant(&other.kind) {
        return false;
    }
    use rustc_ast::token::TokenKind::*;
    match (&first.kind, &other.kind) {
        (&Eq, &Eq)
        | (&Lt, &Lt)
        | (&Le, &Le)
        | (&EqEq, &EqEq)
        | (&Ne, &Ne)
        | (&Ge, &Ge)
        | (&Gt, &Gt)
        | (&AndAnd, &AndAnd)
        | (&OrOr, &OrOr)
        | (&Not, &Not)
        | (&Tilde, &Tilde)
        | (&At, &At)
        | (&Dot, &Dot)
        | (&DotDot, &DotDot)
        | (&DotDotDot, &DotDotDot)
        | (&DotDotEq, &DotDotEq)
        | (&Comma, &Comma)
        | (&Semi, &Semi)
        | (&Colon, &Colon)
        | (&ModSep, &ModSep)
        | (&RArrow, &RArrow)
        | (&LArrow, &LArrow)
        | (&FatArrow, &FatArrow)
        | (&Pound, &Pound)
        | (&Dollar, &Dollar)
        | (&Question, &Question)
        | (&Eof, &Eof) => true,

        (&BinOp(a), &BinOp(b)) | (&BinOpEq(a), &BinOpEq(b)) => a == b,

        (&OpenDelim(a), &OpenDelim(b)) | (&CloseDelim(a), &CloseDelim(b)) => a == b,

        (&DocComment(a1, a2, a3), &DocComment(b1, b2, b3)) => a1 == b1 && a2 == b2 && a3 == b3,

        (&Literal(a), &Literal(b)) => a == b,

        (&Lifetime(a), &Lifetime(b)) => a == b,
        (&Ident(a, b), &Ident(c, d)) => {
            b == d && (a == c || a == kw::DollarCrate || c == kw::DollarCrate)
        }

        (&Interpolated(..), &Interpolated(..)) => panic!("Unexpanded Interpolated!"),

        _ => panic!("forgot to add a token?"),
    }
}

fn prepend_attrs(
    attrs: &[ast::Attribute],
    tokens: Option<&tokenstream::LazyTokenStream>,
) -> Option<tokenstream::TokenStream> {
    let tokens = tokens?.create_token_stream();
    if attrs.is_empty() {
        return Some(tokens);
    }
    let mut builder = tokenstream::TokenStreamBuilder::new();
    for attr in attrs {
        // FIXME: Correctly handle tokens for inner attributes.
        // For now, we fall back to reparsing the original AST node
        if attr.style == ast::AttrStyle::Inner {
            return None;
        }
        builder.push(attr.tokens());
    }
    builder.push(tokens);
    Some(builder.build())
}
