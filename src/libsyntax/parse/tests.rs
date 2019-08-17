use super::*;

use crate::ast::{self, Name, PatKind};
use crate::attr::first_attr_value_str_by_name;
use crate::parse::{ParseSess, PResult};
use crate::parse::new_parser_from_source_str;
use crate::parse::token::Token;
use crate::print::pprust::item_to_string;
use crate::ptr::P;
use crate::source_map::FilePathMapping;
use crate::symbol::{kw, sym};
use crate::tests::{matches_codepattern, string_to_stream, with_error_checking_parse};
use crate::tokenstream::{DelimSpan, TokenTree, TokenStream};
use crate::with_default_globals;
use syntax_pos::{Span, BytePos, Pos};

use std::path::PathBuf;

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
    Span::with_root_ctxt(BytePos(a), BytePos(b))
}

/// Parse a string, return an expr
fn string_to_expr(source_str : String) -> P<ast::Expr> {
    let ps = ParseSess::new(FilePathMapping::empty());
    with_error_checking_parse(source_str, &ps, |p| {
        p.parse_expr()
    })
}

/// Parse a string, return an item
fn string_to_item(source_str : String) -> Option<P<ast::Item>> {
    let ps = ParseSess::new(FilePathMapping::empty());
    with_error_checking_parse(source_str, &ps, |p| {
        p.parse_item()
    })
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
                PatKind::Ident(_ , ref ident, _) => {
                    self.spans.push(ident.span.clone());
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
            ast::ExprKind::Mac(ref mac) => mac.stream().trees().collect(),
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

#[test]
fn eqmodws() {
    assert_eq!(matches_codepattern("",""),true);
    assert_eq!(matches_codepattern("","a"),false);
    assert_eq!(matches_codepattern("a",""),false);
    assert_eq!(matches_codepattern("a","a"),true);
    assert_eq!(matches_codepattern("a b","a   \n\t\r  b"),true);
    assert_eq!(matches_codepattern("a b ","a   \n\t\r  b"),true);
    assert_eq!(matches_codepattern("a b","a   \n\t\r  b "),false);
    assert_eq!(matches_codepattern("a   b","a b"),true);
    assert_eq!(matches_codepattern("ab","a b"),false);
    assert_eq!(matches_codepattern("a   b","ab"),true);
    assert_eq!(matches_codepattern(" a   b","ab"),true);
}

#[test]
fn pattern_whitespace() {
    assert_eq!(matches_codepattern("","\x0C"), false);
    assert_eq!(matches_codepattern("a b ","a   \u{0085}\n\t\r  b"),true);
    assert_eq!(matches_codepattern("a b","a   \u{0085}\n\t\r  b "),false);
}

#[test]
fn non_pattern_whitespace() {
    // These have the property 'White_Space' but not 'Pattern_White_Space'
    assert_eq!(matches_codepattern("a b","a\u{2002}b"), false);
    assert_eq!(matches_codepattern("a   b","a\u{2002}b"), false);
    assert_eq!(matches_codepattern("\u{205F}a   b","ab"), false);
    assert_eq!(matches_codepattern("a  \u{3000}b","ab"), false);
}
