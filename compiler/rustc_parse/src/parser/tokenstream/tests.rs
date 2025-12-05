#![allow(rustc::symbol_intern_string_literal)]

use rustc_ast::token::{self, IdentIsRaw};
use rustc_ast::tokenstream::{TokenStream, TokenTree};
use rustc_span::{BytePos, Span, Symbol, create_default_session_globals_then};

use crate::parser::tests::string_to_stream;

fn string_to_ts(string: &str) -> TokenStream {
    string_to_stream(string.to_owned())
}

fn sp(a: u32, b: u32) -> Span {
    Span::with_root_ctxt(BytePos(a), BytePos(b))
}

fn cmp_token_stream(a: &TokenStream, b: &TokenStream) -> bool {
    a.iter().eq_by(b.iter(), |x, y| x.eq_unspanned(y))
}

#[test]
fn test_concat() {
    create_default_session_globals_then(|| {
        let test_res = string_to_ts("foo::bar::baz");
        let test_fst = string_to_ts("foo::bar");
        let test_snd = string_to_ts("::baz");
        let mut eq_res = TokenStream::default();
        eq_res.push_stream(test_fst);
        eq_res.push_stream(test_snd);
        assert_eq!(test_res.iter().count(), 5);
        assert_eq!(eq_res.iter().count(), 5);
        assert_eq!(cmp_token_stream(&test_res, &eq_res), true);
    })
}

#[test]
fn test_to_from_bijection() {
    create_default_session_globals_then(|| {
        let test_start = string_to_ts("foo::bar(baz)");
        let test_end = test_start.iter().cloned().collect();
        assert_eq!(test_start, test_end)
    })
}

#[test]
fn test_eq_0() {
    create_default_session_globals_then(|| {
        let test_res = string_to_ts("foo");
        let test_eqs = string_to_ts("foo");
        assert_eq!(test_res, test_eqs)
    })
}

#[test]
fn test_eq_1() {
    create_default_session_globals_then(|| {
        let test_res = string_to_ts("::bar::baz");
        let test_eqs = string_to_ts("::bar::baz");
        assert_eq!(test_res, test_eqs)
    })
}

#[test]
fn test_eq_3() {
    create_default_session_globals_then(|| {
        let test_res = string_to_ts("");
        let test_eqs = string_to_ts("");
        assert_eq!(test_res, test_eqs)
    })
}

#[test]
fn test_diseq_0() {
    create_default_session_globals_then(|| {
        let test_res = string_to_ts("::bar::baz");
        let test_eqs = string_to_ts("bar::baz");
        assert_eq!(test_res == test_eqs, false)
    })
}

#[test]
fn test_diseq_1() {
    create_default_session_globals_then(|| {
        let test_res = string_to_ts("(bar,baz)");
        let test_eqs = string_to_ts("bar,baz");
        assert_eq!(test_res == test_eqs, false)
    })
}

#[test]
fn test_is_empty() {
    create_default_session_globals_then(|| {
        let test0 = TokenStream::default();
        let test1 =
            TokenStream::token_alone(token::Ident(Symbol::intern("a"), IdentIsRaw::No), sp(0, 1));
        let test2 = string_to_ts("foo(bar::baz)");

        assert_eq!(test0.is_empty(), true);
        assert_eq!(test1.is_empty(), false);
        assert_eq!(test2.is_empty(), false);
    })
}

#[test]
fn test_dotdotdot() {
    create_default_session_globals_then(|| {
        let mut stream = TokenStream::default();
        stream.push_tree(TokenTree::token_joint(token::Dot, sp(0, 1)));
        stream.push_tree(TokenTree::token_joint(token::Dot, sp(1, 2)));
        stream.push_tree(TokenTree::token_alone(token::Dot, sp(2, 3)));
        assert!(cmp_token_stream(&stream, &string_to_ts("...")));
        assert_eq!(stream.iter().count(), 1);
    })
}
