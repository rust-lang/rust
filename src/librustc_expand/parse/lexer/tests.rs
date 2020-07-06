use rustc_ast::token::{self, Token, TokenKind};
use rustc_ast::util::comments::is_doc_comment;
use rustc_ast::with_default_session_globals;
use rustc_data_structures::sync::Lrc;
use rustc_errors::{emitter::EmitterWriter, Handler};
use rustc_parse::lexer::StringReader;
use rustc_session::parse::ParseSess;
use rustc_span::source_map::{FilePathMapping, SourceMap};
use rustc_span::symbol::Symbol;
use rustc_span::{BytePos, Span};

use std::io;
use std::path::PathBuf;

fn mk_sess(sm: Lrc<SourceMap>) -> ParseSess {
    let emitter = EmitterWriter::new(
        Box::new(io::sink()),
        Some(sm.clone()),
        false,
        false,
        false,
        None,
        false,
    );
    ParseSess::with_span_handler(Handler::with_emitter(true, None, Box::new(emitter)), sm)
}

// Creates a string reader for the given string.
fn setup<'a>(sm: &SourceMap, sess: &'a ParseSess, teststr: String) -> StringReader<'a> {
    let sf = sm.new_source_file(PathBuf::from(teststr.clone()).into(), teststr);
    StringReader::new(sess, sf, None)
}

#[test]
fn t1() {
    with_default_session_globals(|| {
        let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
        let sh = mk_sess(sm.clone());
        let mut string_reader = setup(
            &sm,
            &sh,
            "/* my source file */ fn main() { println!(\"zebra\"); }\n".to_string(),
        );
        assert_eq!(string_reader.next_token(), token::Comment);
        assert_eq!(string_reader.next_token(), token::Whitespace);
        let tok1 = string_reader.next_token();
        let tok2 = Token::new(mk_ident("fn"), Span::with_root_ctxt(BytePos(21), BytePos(23)));
        assert_eq!(tok1.kind, tok2.kind);
        assert_eq!(tok1.span, tok2.span);
        assert_eq!(string_reader.next_token(), token::Whitespace);
        // Read another token.
        let tok3 = string_reader.next_token();
        assert_eq!(string_reader.pos(), BytePos(28));
        let tok4 = Token::new(mk_ident("main"), Span::with_root_ctxt(BytePos(24), BytePos(28)));
        assert_eq!(tok3.kind, tok4.kind);
        assert_eq!(tok3.span, tok4.span);

        assert_eq!(string_reader.next_token(), token::OpenDelim(token::Paren));
        assert_eq!(string_reader.pos(), BytePos(29))
    })
}

// Checks that the given reader produces the desired stream
// of tokens (stop checking after exhausting `expected`).
fn check_tokenization(mut string_reader: StringReader<'_>, expected: Vec<TokenKind>) {
    for expected_tok in &expected {
        assert_eq!(&string_reader.next_token(), expected_tok);
    }
}

// Makes the identifier by looking up the string in the interner.
fn mk_ident(id: &str) -> TokenKind {
    token::Ident(Symbol::intern(id), false)
}

fn mk_lit(kind: token::LitKind, symbol: &str, suffix: Option<&str>) -> TokenKind {
    TokenKind::lit(kind, Symbol::intern(symbol), suffix.map(Symbol::intern))
}

#[test]
fn doublecolon_parsing() {
    with_default_session_globals(|| {
        let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
        let sh = mk_sess(sm.clone());
        check_tokenization(
            setup(&sm, &sh, "a b".to_string()),
            vec![mk_ident("a"), token::Whitespace, mk_ident("b")],
        );
    })
}

#[test]
fn doublecolon_parsing_2() {
    with_default_session_globals(|| {
        let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
        let sh = mk_sess(sm.clone());
        check_tokenization(
            setup(&sm, &sh, "a::b".to_string()),
            vec![mk_ident("a"), token::Colon, token::Colon, mk_ident("b")],
        );
    })
}

#[test]
fn doublecolon_parsing_3() {
    with_default_session_globals(|| {
        let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
        let sh = mk_sess(sm.clone());
        check_tokenization(
            setup(&sm, &sh, "a ::b".to_string()),
            vec![mk_ident("a"), token::Whitespace, token::Colon, token::Colon, mk_ident("b")],
        );
    })
}

#[test]
fn doublecolon_parsing_4() {
    with_default_session_globals(|| {
        let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
        let sh = mk_sess(sm.clone());
        check_tokenization(
            setup(&sm, &sh, "a:: b".to_string()),
            vec![mk_ident("a"), token::Colon, token::Colon, token::Whitespace, mk_ident("b")],
        );
    })
}

#[test]
fn character_a() {
    with_default_session_globals(|| {
        let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
        let sh = mk_sess(sm.clone());
        assert_eq!(setup(&sm, &sh, "'a'".to_string()).next_token(), mk_lit(token::Char, "a", None),);
    })
}

#[test]
fn character_space() {
    with_default_session_globals(|| {
        let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
        let sh = mk_sess(sm.clone());
        assert_eq!(setup(&sm, &sh, "' '".to_string()).next_token(), mk_lit(token::Char, " ", None),);
    })
}

#[test]
fn character_escaped() {
    with_default_session_globals(|| {
        let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
        let sh = mk_sess(sm.clone());
        assert_eq!(
            setup(&sm, &sh, "'\\n'".to_string()).next_token(),
            mk_lit(token::Char, "\\n", None),
        );
    })
}

#[test]
fn lifetime_name() {
    with_default_session_globals(|| {
        let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
        let sh = mk_sess(sm.clone());
        assert_eq!(
            setup(&sm, &sh, "'abc".to_string()).next_token(),
            token::Lifetime(Symbol::intern("'abc")),
        );
    })
}

#[test]
fn raw_string() {
    with_default_session_globals(|| {
        let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
        let sh = mk_sess(sm.clone());
        assert_eq!(
            setup(&sm, &sh, "r###\"\"#a\\b\x00c\"\"###".to_string()).next_token(),
            mk_lit(token::StrRaw(3), "\"#a\\b\x00c\"", None),
        );
    })
}

#[test]
fn literal_suffixes() {
    with_default_session_globals(|| {
        let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
        let sh = mk_sess(sm.clone());
        macro_rules! test {
            ($input: expr, $tok_type: ident, $tok_contents: expr) => {{
                assert_eq!(
                    setup(&sm, &sh, format!("{}suffix", $input)).next_token(),
                    mk_lit(token::$tok_type, $tok_contents, Some("suffix")),
                );
                // with a whitespace separator
                assert_eq!(
                    setup(&sm, &sh, format!("{} suffix", $input)).next_token(),
                    mk_lit(token::$tok_type, $tok_contents, None),
                );
            }};
        }

        test!("'a'", Char, "a");
        test!("b'a'", Byte, "a");
        test!("\"a\"", Str, "a");
        test!("b\"a\"", ByteStr, "a");
        test!("1234", Integer, "1234");
        test!("0b101", Integer, "0b101");
        test!("0xABC", Integer, "0xABC");
        test!("1.0", Float, "1.0");
        test!("1.0e10", Float, "1.0e10");

        assert_eq!(
            setup(&sm, &sh, "2us".to_string()).next_token(),
            mk_lit(token::Integer, "2", Some("us")),
        );
        assert_eq!(
            setup(&sm, &sh, "r###\"raw\"###suffix".to_string()).next_token(),
            mk_lit(token::StrRaw(3), "raw", Some("suffix")),
        );
        assert_eq!(
            setup(&sm, &sh, "br###\"raw\"###suffix".to_string()).next_token(),
            mk_lit(token::ByteStrRaw(3), "raw", Some("suffix")),
        );
    })
}

#[test]
fn line_doc_comments() {
    assert!(is_doc_comment("///"));
    assert!(is_doc_comment("/// blah"));
    assert!(!is_doc_comment("////"));
}

#[test]
fn nested_block_comments() {
    with_default_session_globals(|| {
        let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
        let sh = mk_sess(sm.clone());
        let mut lexer = setup(&sm, &sh, "/* /* */ */'a'".to_string());
        assert_eq!(lexer.next_token(), token::Comment);
        assert_eq!(lexer.next_token(), mk_lit(token::Char, "a", None));
    })
}

#[test]
fn crlf_comments() {
    with_default_session_globals(|| {
        let sm = Lrc::new(SourceMap::new(FilePathMapping::empty()));
        let sh = mk_sess(sm.clone());
        let mut lexer = setup(&sm, &sh, "// test\r\n/// test\r\n".to_string());
        let comment = lexer.next_token();
        assert_eq!(comment.kind, token::Comment);
        assert_eq!((comment.span.lo(), comment.span.hi()), (BytePos(0), BytePos(7)));
        assert_eq!(lexer.next_token(), token::Whitespace);
        assert_eq!(lexer.next_token(), token::DocComment(Symbol::intern("/// test")));
    })
}
