//! The main parser interface

use crate::ast::{self, CrateConfig, NodeId};
use crate::early_buffered_lints::{BufferedEarlyLint, BufferedEarlyLintId};
use crate::source_map::{SourceMap, FilePathMapping};
use crate::errors::{FatalError, Level, Handler, ColorConfig, Diagnostic, DiagnosticBuilder};
use crate::feature_gate::UnstableFeatures;
use crate::parse::parser::Parser;
use crate::symbol::Symbol;
use crate::tokenstream::{TokenStream, TokenTree};
use crate::diagnostics::plugin::ErrorMap;

use rustc_data_structures::sync::{Lrc, Lock};
use syntax_pos::{Span, SourceFile, FileName, MultiSpan};
use log::debug;

use rustc_data_structures::fx::FxHashSet;
use std::borrow::Cow;
use std::iter;
use std::path::{Path, PathBuf};
use std::str;

pub type PResult<'a, T> = Result<T, DiagnosticBuilder<'a>>;

#[macro_use]
pub mod parser;

pub mod lexer;
pub mod token;
pub mod attr;

pub mod classify;

/// Info about a parsing session.
pub struct ParseSess {
    pub span_diagnostic: Handler,
    pub unstable_features: UnstableFeatures,
    pub config: CrateConfig,
    pub missing_fragment_specifiers: Lock<FxHashSet<Span>>,
    /// Places where raw identifiers were used. This is used for feature gating
    /// raw identifiers
    pub raw_identifier_spans: Lock<Vec<Span>>,
    /// The registered diagnostics codes
    crate registered_diagnostics: Lock<ErrorMap>,
    /// Used to determine and report recursive mod inclusions
    included_mod_stack: Lock<Vec<PathBuf>>,
    source_map: Lrc<SourceMap>,
    pub buffered_lints: Lock<Vec<BufferedEarlyLint>>,
}

impl ParseSess {
    pub fn new(file_path_mapping: FilePathMapping) -> Self {
        let cm = Lrc::new(SourceMap::new(file_path_mapping));
        let handler = Handler::with_tty_emitter(ColorConfig::Auto,
                                                true,
                                                false,
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

pub fn parse_stream_from_source_str(name: FileName, source: String, sess: &ParseSess,
                                    override_span: Option<Span>)
                                    -> TokenStream {
    source_file_to_stream(sess, sess.source_map().new_source_file(name, source), override_span)
}

/// Create a new parser from a source string
pub fn new_parser_from_source_str(sess: &ParseSess, name: FileName, source: String)
                                      -> Parser<'_> {
    panictry_buffer!(&sess.span_diagnostic, maybe_new_parser_from_source_str(sess, name, source))
}

/// Create a new parser from a source string. Returns any buffered errors from lexing the initial
/// token stream.
pub fn maybe_new_parser_from_source_str(sess: &ParseSess, name: FileName, source: String)
    -> Result<Parser<'_>, Vec<Diagnostic>>
{
    let mut parser = maybe_source_file_to_parser(sess,
                                                 sess.source_map().new_source_file(name, source))?;
    parser.recurse_into_file_modules = false;
    Ok(parser)
}

/// Create a new parser, handling errors as appropriate
/// if the file doesn't exist
pub fn new_parser_from_file<'a>(sess: &'a ParseSess, path: &Path) -> Parser<'a> {
    source_file_to_parser(sess, file_to_source_file(sess, path, None))
}

/// Create a new parser, returning buffered diagnostics if the file doesn't
/// exist or from lexing the initial token stream.
pub fn maybe_new_parser_from_file<'a>(sess: &'a ParseSess, path: &Path)
    -> Result<Parser<'a>, Vec<Diagnostic>> {
    let file = try_file_to_source_file(sess, path, None).map_err(|db| vec![db])?;
    maybe_source_file_to_parser(sess, file)
}

/// Given a session, a crate config, a path, and a span, add
/// the file at the given path to the source_map, and return a parser.
/// On an error, use the given span as the source of the problem.
crate fn new_sub_parser_from_file<'a>(sess: &'a ParseSess,
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
fn maybe_source_file_to_parser(sess: &ParseSess, source_file: Lrc<SourceFile>)
    -> Result<Parser<'_>, Vec<Diagnostic>>
{
    let end_pos = source_file.end_pos;
    let mut parser = stream_to_parser(sess, maybe_file_to_stream(sess, source_file, None)?);

    if parser.token == token::Eof && parser.span.is_dummy() {
        parser.span = Span::new(end_pos, end_pos, parser.span.ctxt());
    }

    Ok(parser)
}

// must preserve old name for now, because quote! from the *existing*
// compiler expands into it
pub fn new_parser_from_tts(sess: &ParseSess, tts: Vec<TokenTree>) -> Parser<'_> {
    stream_to_parser(sess, tts.into_iter().collect())
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
/// add the path to the session's source_map and return the new source_file.
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

/// Given a source_file, produce a sequence of token-trees
pub fn source_file_to_stream(sess: &ParseSess,
                             source_file: Lrc<SourceFile>,
                             override_span: Option<Span>) -> TokenStream {
    panictry_buffer!(&sess.span_diagnostic, maybe_file_to_stream(sess, source_file, override_span))
}

/// Given a source file, produce a sequence of token-trees. Returns any buffered errors from
/// parsing the token tream.
pub fn maybe_file_to_stream(sess: &ParseSess,
                            source_file: Lrc<SourceFile>,
                            override_span: Option<Span>) -> Result<TokenStream, Vec<Diagnostic>> {
    let mut srdr = lexer::StringReader::new_or_buffered_errs(sess, source_file, override_span)?;
    srdr.real_token();

    match srdr.parse_all_token_trees() {
        Ok(stream) => Ok(stream),
        Err(err) => {
            let mut buffer = Vec::with_capacity(1);
            err.buffer(&mut buffer);
            Err(buffer)
        }
    }
}

/// Given stream and the `ParseSess`, produce a parser
pub fn stream_to_parser(sess: &ParseSess, stream: TokenStream) -> Parser<'_> {
    Parser::new(sess, stream, None, true, false)
}

/// Parse a string representing a character literal into its final form.
/// Rather than just accepting/rejecting a given literal, unescapes it as
/// well. Can take any slice prefixed by a character escape. Returns the
/// character and the number of characters consumed.
fn char_lit(lit: &str, diag: Option<(Span, &Handler)>) -> (char, isize) {
    use std::char;

    // Handle non-escaped chars first.
    if lit.as_bytes()[0] != b'\\' {
        // If the first byte isn't '\\' it might part of a multi-byte char, so
        // get the char with chars().
        let c = lit.chars().next().unwrap();
        return (c, 1);
    }

    // Handle escaped chars.
    match lit.as_bytes()[1] as char {
        '"' => ('"', 2),
        'n' => ('\n', 2),
        'r' => ('\r', 2),
        't' => ('\t', 2),
        '\\' => ('\\', 2),
        '\'' => ('\'', 2),
        '0' => ('\0', 2),
        'x' => {
            let v = u32::from_str_radix(&lit[2..4], 16).unwrap();
            let c = char::from_u32(v).unwrap();
            (c, 4)
        }
        'u' => {
            assert_eq!(lit.as_bytes()[2], b'{');
            let idx = lit.find('}').unwrap();

            // All digits and '_' are ascii, so treat each byte as a char.
            let mut v: u32 = 0;
            for c in lit[3..idx].bytes() {
                let c = char::from(c);
                if c != '_' {
                    let x = c.to_digit(16).unwrap();
                    v = v.checked_mul(16).unwrap().checked_add(x).unwrap();
                }
            }
            let c = char::from_u32(v).unwrap_or_else(|| {
                if let Some((span, diag)) = diag {
                    let mut diag = diag.struct_span_err(span, "invalid unicode character escape");
                    if v > 0x10FFFF {
                        diag.help("unicode escape must be at most 10FFFF").emit();
                    } else {
                        diag.help("unicode escape must not be a surrogate").emit();
                    }
                }
                '\u{FFFD}'
            });
            (c, (idx + 1) as isize)
        }
        _ => panic!("lexer should have rejected a bad character escape {}", lit)
    }
}

/// Parse a string representing a string literal into its final form. Does
/// unescaping.
pub fn str_lit(lit: &str, diag: Option<(Span, &Handler)>) -> String {
    debug!("str_lit: given {}", lit.escape_default());
    let mut res = String::with_capacity(lit.len());

    let error = |i| format!("lexer should have rejected {} at {}", lit, i);

    /// Eat everything up to a non-whitespace
    fn eat<'a>(it: &mut iter::Peekable<str::CharIndices<'a>>) {
        loop {
            match it.peek().map(|x| x.1) {
                Some(' ') | Some('\n') | Some('\r') | Some('\t') => {
                    it.next();
                },
                _ => { break; }
            }
        }
    }

    let mut chars = lit.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        match c {
            '\\' => {
                let ch = chars.peek().unwrap_or_else(|| {
                    panic!("{}", error(i))
                }).1;

                if ch == '\n' {
                    eat(&mut chars);
                } else if ch == '\r' {
                    chars.next();
                    let ch = chars.peek().unwrap_or_else(|| {
                        panic!("{}", error(i))
                    }).1;

                    if ch != '\n' {
                        panic!("lexer accepted bare CR");
                    }
                    eat(&mut chars);
                } else {
                    // otherwise, a normal escape
                    let (c, n) = char_lit(&lit[i..], diag);
                    for _ in 0..n - 1 { // we don't need to move past the first \
                        chars.next();
                    }
                    res.push(c);
                }
            },
            '\r' => {
                let ch = chars.peek().unwrap_or_else(|| {
                    panic!("{}", error(i))
                }).1;

                if ch != '\n' {
                    panic!("lexer accepted bare CR");
                }
                chars.next();
                res.push('\n');
            }
            c => res.push(c),
        }
    }

    res.shrink_to_fit(); // probably not going to do anything, unless there was an escape.
    debug!("parse_str_lit: returning {}", res);
    res
}

/// Parse a string representing a raw string literal into its final form. The
/// only operation this does is convert embedded CRLF into a single LF.
fn raw_str_lit(lit: &str) -> String {
    debug!("raw_str_lit: given {}", lit.escape_default());
    let mut res = String::with_capacity(lit.len());

    let mut chars = lit.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\r' {
            if *chars.peek().unwrap() != '\n' {
                panic!("lexer accepted bare CR");
            }
            chars.next();
            res.push('\n');
        } else {
            res.push(c);
        }
    }

    res.shrink_to_fit();
    res
}

// check if `s` looks like i32 or u1234 etc.
fn looks_like_width_suffix(first_chars: &[char], s: &str) -> bool {
    s.starts_with(first_chars) && s[1..].chars().all(|c| c.is_ascii_digit())
}

macro_rules! err {
    ($opt_diag:expr, |$span:ident, $diag:ident| $($body:tt)*) => {
        match $opt_diag {
            Some(($span, $diag)) => { $($body)* }
            None => return None,
        }
    }
}

crate fn lit_token(lit: token::Lit, suf: Option<Symbol>, diag: Option<(Span, &Handler)>)
                 -> (bool /* suffix illegal? */, Option<ast::LitKind>) {
    use ast::LitKind;

    match lit {
       token::Byte(i) => (true, Some(LitKind::Byte(byte_lit(&i.as_str()).0))),
       token::Char(i) => (true, Some(LitKind::Char(char_lit(&i.as_str(), diag).0))),
       token::Err(i) => (true, Some(LitKind::Err(i))),

        // There are some valid suffixes for integer and float literals,
        // so all the handling is done internally.
        token::Integer(s) => (false, integer_lit(&s.as_str(), suf, diag)),
        token::Float(s) => (false, float_lit(&s.as_str(), suf, diag)),

        token::Str_(mut sym) => {
            // If there are no characters requiring special treatment we can
            // reuse the symbol from the Token. Otherwise, we must generate a
            // new symbol because the string in the LitKind is different to the
            // string in the Token.
            let s = &sym.as_str();
            if s.as_bytes().iter().any(|&c| c == b'\\' || c == b'\r') {
                sym = Symbol::intern(&str_lit(s, diag));
            }
            (true, Some(LitKind::Str(sym, ast::StrStyle::Cooked)))
        }
        token::StrRaw(mut sym, n) => {
            // Ditto.
            let s = &sym.as_str();
            if s.contains('\r') {
                sym = Symbol::intern(&raw_str_lit(s));
            }
            (true, Some(LitKind::Str(sym, ast::StrStyle::Raw(n))))
        }
        token::ByteStr(i) => {
            (true, Some(LitKind::ByteStr(byte_str_lit(&i.as_str()))))
        }
        token::ByteStrRaw(i, _) => {
            (true, Some(LitKind::ByteStr(Lrc::new(i.to_string().into_bytes()))))
        }
    }
}

fn filtered_float_lit(data: Symbol, suffix: Option<Symbol>, diag: Option<(Span, &Handler)>)
                      -> Option<ast::LitKind> {
    debug!("filtered_float_lit: {}, {:?}", data, suffix);
    let suffix = match suffix {
        Some(suffix) => suffix,
        None => return Some(ast::LitKind::FloatUnsuffixed(data)),
    };

    Some(match &*suffix.as_str() {
        "f32" => ast::LitKind::Float(data, ast::FloatTy::F32),
        "f64" => ast::LitKind::Float(data, ast::FloatTy::F64),
        suf => {
            err!(diag, |span, diag| {
                if suf.len() >= 2 && looks_like_width_suffix(&['f'], suf) {
                    // if it looks like a width, lets try to be helpful.
                    let msg = format!("invalid width `{}` for float literal", &suf[1..]);
                    diag.struct_span_err(span, &msg).help("valid widths are 32 and 64").emit()
                } else {
                    let msg = format!("invalid suffix `{}` for float literal", suf);
                    diag.struct_span_err(span, &msg)
                        .span_label(span, format!("invalid suffix `{}`", suf))
                        .help("valid suffixes are `f32` and `f64`")
                        .emit();
                }
            });

            ast::LitKind::FloatUnsuffixed(data)
        }
    })
}
fn float_lit(s: &str, suffix: Option<Symbol>, diag: Option<(Span, &Handler)>)
                 -> Option<ast::LitKind> {
    debug!("float_lit: {:?}, {:?}", s, suffix);
    // FIXME #2252: bounds checking float literals is deferred until trans

    // Strip underscores without allocating a new String unless necessary.
    let s2;
    let s = if s.chars().any(|c| c == '_') {
        s2 = s.chars().filter(|&c| c != '_').collect::<String>();
        &s2
    } else {
        s
    };

    filtered_float_lit(Symbol::intern(s), suffix, diag)
}

/// Parse a string representing a byte literal into its final form. Similar to `char_lit`
fn byte_lit(lit: &str) -> (u8, usize) {
    let err = |i| format!("lexer accepted invalid byte literal {} step {}", lit, i);

    if lit.len() == 1 {
        (lit.as_bytes()[0], 1)
    } else {
        assert_eq!(lit.as_bytes()[0], b'\\', "{}", err(0));
        let b = match lit.as_bytes()[1] {
            b'"' => b'"',
            b'n' => b'\n',
            b'r' => b'\r',
            b't' => b'\t',
            b'\\' => b'\\',
            b'\'' => b'\'',
            b'0' => b'\0',
            _ => {
                match u64::from_str_radix(&lit[2..4], 16).ok() {
                    Some(c) =>
                        if c > 0xFF {
                            panic!(err(2))
                        } else {
                            return (c as u8, 4)
                        },
                    None => panic!(err(3))
                }
            }
        };
        (b, 2)
    }
}

fn byte_str_lit(lit: &str) -> Lrc<Vec<u8>> {
    let mut res = Vec::with_capacity(lit.len());

    let error = |i| panic!("lexer should have rejected {} at {}", lit, i);

    /// Eat everything up to a non-whitespace
    fn eat<I: Iterator<Item=(usize, u8)>>(it: &mut iter::Peekable<I>) {
        loop {
            match it.peek().map(|x| x.1) {
                Some(b' ') | Some(b'\n') | Some(b'\r') | Some(b'\t') => {
                    it.next();
                },
                _ => { break; }
            }
        }
    }

    // byte string literals *must* be ASCII, but the escapes don't have to be
    let mut chars = lit.bytes().enumerate().peekable();
    loop {
        match chars.next() {
            Some((i, b'\\')) => {
                match chars.peek().unwrap_or_else(|| error(i)).1 {
                    b'\n' => eat(&mut chars),
                    b'\r' => {
                        chars.next();
                        if chars.peek().unwrap_or_else(|| error(i)).1 != b'\n' {
                            panic!("lexer accepted bare CR");
                        }
                        eat(&mut chars);
                    }
                    _ => {
                        // otherwise, a normal escape
                        let (c, n) = byte_lit(&lit[i..]);
                        // we don't need to move past the first \
                        for _ in 0..n - 1 {
                            chars.next();
                        }
                        res.push(c);
                    }
                }
            },
            Some((i, b'\r')) => {
                if chars.peek().unwrap_or_else(|| error(i)).1 != b'\n' {
                    panic!("lexer accepted bare CR");
                }
                chars.next();
                res.push(b'\n');
            }
            Some((_, c)) => res.push(c),
            None => break,
        }
    }

    Lrc::new(res)
}

fn integer_lit(s: &str, suffix: Option<Symbol>, diag: Option<(Span, &Handler)>)
                   -> Option<ast::LitKind> {
    // s can only be ascii, byte indexing is fine

    // Strip underscores without allocating a new String unless necessary.
    let s2;
    let mut s = if s.chars().any(|c| c == '_') {
        s2 = s.chars().filter(|&c| c != '_').collect::<String>();
        &s2
    } else {
        s
    };

    debug!("integer_lit: {}, {:?}", s, suffix);

    let mut base = 10;
    let orig = s;
    let mut ty = ast::LitIntType::Unsuffixed;

    if s.starts_with('0') && s.len() > 1 {
        match s.as_bytes()[1] {
            b'x' => base = 16,
            b'o' => base = 8,
            b'b' => base = 2,
            _ => { }
        }
    }

    // 1f64 and 2f32 etc. are valid float literals.
    if let Some(suf) = suffix {
        if looks_like_width_suffix(&['f'], &suf.as_str()) {
            let err = match base {
                16 => Some("hexadecimal float literal is not supported"),
                8 => Some("octal float literal is not supported"),
                2 => Some("binary float literal is not supported"),
                _ => None,
            };
            if let Some(err) = err {
                err!(diag, |span, diag| {
                    diag.struct_span_err(span, err)
                        .span_label(span, "not supported")
                        .emit();
                });
            }
            return filtered_float_lit(Symbol::intern(s), Some(suf), diag)
        }
    }

    if base != 10 {
        s = &s[2..];
    }

    if let Some(suf) = suffix {
        if suf.as_str().is_empty() {
            err!(diag, |span, diag| diag.span_bug(span, "found empty literal suffix in Some"));
        }
        ty = match &*suf.as_str() {
            "isize" => ast::LitIntType::Signed(ast::IntTy::Isize),
            "i8"  => ast::LitIntType::Signed(ast::IntTy::I8),
            "i16" => ast::LitIntType::Signed(ast::IntTy::I16),
            "i32" => ast::LitIntType::Signed(ast::IntTy::I32),
            "i64" => ast::LitIntType::Signed(ast::IntTy::I64),
            "i128" => ast::LitIntType::Signed(ast::IntTy::I128),
            "usize" => ast::LitIntType::Unsigned(ast::UintTy::Usize),
            "u8"  => ast::LitIntType::Unsigned(ast::UintTy::U8),
            "u16" => ast::LitIntType::Unsigned(ast::UintTy::U16),
            "u32" => ast::LitIntType::Unsigned(ast::UintTy::U32),
            "u64" => ast::LitIntType::Unsigned(ast::UintTy::U64),
            "u128" => ast::LitIntType::Unsigned(ast::UintTy::U128),
            suf => {
                // i<digits> and u<digits> look like widths, so lets
                // give an error message along those lines
                err!(diag, |span, diag| {
                    if looks_like_width_suffix(&['i', 'u'], suf) {
                        let msg = format!("invalid width `{}` for integer literal", &suf[1..]);
                        diag.struct_span_err(span, &msg)
                            .help("valid widths are 8, 16, 32, 64 and 128")
                            .emit();
                    } else {
                        let msg = format!("invalid suffix `{}` for numeric literal", suf);
                        diag.struct_span_err(span, &msg)
                            .span_label(span, format!("invalid suffix `{}`", suf))
                            .help("the suffix must be one of the integral types \
                                   (`u32`, `isize`, etc)")
                            .emit();
                    }
                });

                ty
            }
        }
    }

    debug!("integer_lit: the type is {:?}, base {:?}, the new string is {:?}, the original \
           string was {:?}, the original suffix was {:?}", ty, base, s, orig, suffix);

    Some(match u128::from_str_radix(s, base) {
        Ok(r) => ast::LitKind::Int(r, ty),
        Err(_) => {
            // small bases are lexed as if they were base 10, e.g, the string
            // might be `0b10201`. This will cause the conversion above to fail,
            // but these cases have errors in the lexer: we don't want to emit
            // two errors, and we especially don't want to emit this error since
            // it isn't necessarily true.
            let already_errored = base < 10 &&
                s.chars().any(|c| c.to_digit(10).map_or(false, |d| d >= base));

            if !already_errored {
                err!(diag, |span, diag| diag.span_err(span, "int literal is too large"));
            }
            ast::LitKind::Int(0, ty)
        }
    })
}

/// `SeqSep` : a sequence separator (token)
/// and whether a trailing separator is allowed.
pub struct SeqSep {
    pub sep: Option<token::Token>,
    pub trailing_sep_allowed: bool,
}

impl SeqSep {
    pub fn trailing_allowed(t: token::Token) -> SeqSep {
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
    use crate::ast::{self, Ident, PatKind};
    use crate::attr::first_attr_value_str_by_name;
    use crate::ptr::P;
    use crate::print::pprust::item_to_string;
    use crate::tokenstream::{DelimSpan, TokenTree};
    use crate::util::parser_testing::string_to_stream;
    use crate::util::parser_testing::{string_to_expr, string_to_item};
    use crate::with_globals;
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
        with_globals(|| {
            string_to_expr("::abc::def::return".to_string());
        })
    }

    // check the token-tree-ization of macros
    #[test]
    fn string_to_tts_macro () {
        with_globals(|| {
            let tts: Vec<_> =
                string_to_stream("macro_rules! zip (($a)=>($a))".to_string()).trees().collect();
            let tts: &[TokenTree] = &tts[..];

            match (tts.len(), tts.get(0), tts.get(1), tts.get(2), tts.get(3)) {
                (
                    4,
                    Some(&TokenTree::Token(_, token::Ident(name_macro_rules, false))),
                    Some(&TokenTree::Token(_, token::Not)),
                    Some(&TokenTree::Token(_, token::Ident(name_zip, false))),
                    Some(&TokenTree::Delimited(_, macro_delim, ref macro_tts)),
                )
                if name_macro_rules.name == "macro_rules"
                && name_zip.name == "zip" => {
                    let tts = &macro_tts.trees().collect::<Vec<_>>();
                    match (tts.len(), tts.get(0), tts.get(1), tts.get(2)) {
                        (
                            3,
                            Some(&TokenTree::Delimited(_, first_delim, ref first_tts)),
                            Some(&TokenTree::Token(_, token::FatArrow)),
                            Some(&TokenTree::Delimited(_, second_delim, ref second_tts)),
                        )
                        if macro_delim == token::Paren => {
                            let tts = &first_tts.trees().collect::<Vec<_>>();
                            match (tts.len(), tts.get(0), tts.get(1)) {
                                (
                                    2,
                                    Some(&TokenTree::Token(_, token::Dollar)),
                                    Some(&TokenTree::Token(_, token::Ident(ident, false))),
                                )
                                if first_delim == token::Paren && ident.name == "a" => {},
                                _ => panic!("value 3: {:?} {:?}", first_delim, first_tts),
                            }
                            let tts = &second_tts.trees().collect::<Vec<_>>();
                            match (tts.len(), tts.get(0), tts.get(1)) {
                                (
                                    2,
                                    Some(&TokenTree::Token(_, token::Dollar)),
                                    Some(&TokenTree::Token(_, token::Ident(ident, false))),
                                )
                                if second_delim == token::Paren && ident.name == "a" => {},
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
        with_globals(|| {
            let tts = string_to_stream("fn a (b : i32) { b; }".to_string());

            let expected = TokenStream::new(vec![
                TokenTree::Token(sp(0, 2), token::Ident(Ident::from_str("fn"), false)).into(),
                TokenTree::Token(sp(3, 4), token::Ident(Ident::from_str("a"), false)).into(),
                TokenTree::Delimited(
                    DelimSpan::from_pair(sp(5, 6), sp(13, 14)),
                    token::DelimToken::Paren,
                    TokenStream::new(vec![
                        TokenTree::Token(sp(6, 7),
                                         token::Ident(Ident::from_str("b"), false)).into(),
                        TokenTree::Token(sp(8, 9), token::Colon).into(),
                        TokenTree::Token(sp(10, 13),
                                         token::Ident(Ident::from_str("i32"), false)).into(),
                    ]).into(),
                ).into(),
                TokenTree::Delimited(
                    DelimSpan::from_pair(sp(15, 16), sp(20, 21)),
                    token::DelimToken::Brace,
                    TokenStream::new(vec![
                        TokenTree::Token(sp(17, 18),
                                         token::Ident(Ident::from_str("b"), false)).into(),
                        TokenTree::Token(sp(18, 19), token::Semi).into(),
                    ]).into(),
                ).into()
            ]);

            assert_eq!(tts, expected);
        })
    }

    #[test] fn parse_use() {
        with_globals(|| {
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
        with_globals(|| {
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
        with_globals(|| {

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
        with_globals(|| {
            // just make sure that they parse....
            string_to_expr("3 + 4".to_string());
            string_to_expr("a::z.froob(b,&(987+3))".to_string());
        })
    }

    #[test] fn attrs_fix_bug () {
        with_globals(|| {
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
        with_globals(|| {
            let sess = ParseSess::new(FilePathMapping::empty());

            let name_1 = FileName::Custom("crlf_source_1".to_string());
            let source = "/// doc comment\r\nfn foo() {}".to_string();
            let item = parse_item_from_source_str(name_1, source, &sess)
                .unwrap().unwrap();
            let doc = first_attr_value_str_by_name(&item.attrs, "doc").unwrap();
            assert_eq!(doc, "/// doc comment");

            let name_2 = FileName::Custom("crlf_source_2".to_string());
            let source = "/// doc comment\r\n/// line 2\r\nfn foo() {}".to_string();
            let item = parse_item_from_source_str(name_2, source, &sess)
                .unwrap().unwrap();
            let docs = item.attrs.iter().filter(|a| a.path == "doc")
                        .map(|a| a.value_str().unwrap().to_string()).collect::<Vec<_>>();
            let b: &[_] = &["/// doc comment".to_string(), "/// line 2".to_string()];
            assert_eq!(&docs[..], b);

            let name_3 = FileName::Custom("clrf_source_3".to_string());
            let source = "/** doc comment\r\n *  with CRLF */\r\nfn foo() {}".to_string();
            let item = parse_item_from_source_str(name_3, source, &sess).unwrap().unwrap();
            let doc = first_attr_value_str_by_name(&item.attrs, "doc").unwrap();
            assert_eq!(doc, "/** doc comment\n *  with CRLF */");
        });
    }

    #[test]
    fn ttdelim_span() {
        fn parse_expr_from_source_str(
            name: FileName, source: String, sess: &ParseSess
        ) -> PResult<'_, P<ast::Expr>> {
            new_parser_from_source_str(sess, name, source).parse_expr()
        }

        with_globals(|| {
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
        with_globals(|| {
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
