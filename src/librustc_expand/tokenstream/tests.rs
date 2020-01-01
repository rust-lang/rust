use crate::tests::string_to_stream;

use rustc_span::{BytePos, Span};
use smallvec::smallvec;
use syntax::ast::Name;
use syntax::token;
use syntax::tokenstream::{TokenStream, TokenStreamBuilder, TokenTree};
use syntax::with_default_globals;

fn string_to_ts(string: &str) -> TokenStream {
    string_to_stream(string.to_owned())
}

fn sp(a: u32, b: u32) -> Span {
    Span::with_root_ctxt(BytePos(a), BytePos(b))
}

#[test]
fn test_concat() {
    with_default_globals(|| {
        let test_res = string_to_ts("foo::bar::baz");
        let test_fst = string_to_ts("foo::bar");
        let test_snd = string_to_ts("::baz");
        let eq_res = TokenStream::from_streams(smallvec![test_fst, test_snd]);
        assert_eq!(test_res.trees().count(), 5);
        assert_eq!(eq_res.trees().count(), 5);
        assert_eq!(test_res.eq_unspanned(&eq_res), true);
    })
}

#[test]
fn test_to_from_bijection() {
    with_default_globals(|| {
        let test_start = string_to_ts("foo::bar(baz)");
        let test_end = test_start.trees().collect();
        assert_eq!(test_start, test_end)
    })
}

#[test]
fn test_eq_0() {
    with_default_globals(|| {
        let test_res = string_to_ts("foo");
        let test_eqs = string_to_ts("foo");
        assert_eq!(test_res, test_eqs)
    })
}

#[test]
fn test_eq_1() {
    with_default_globals(|| {
        let test_res = string_to_ts("::bar::baz");
        let test_eqs = string_to_ts("::bar::baz");
        assert_eq!(test_res, test_eqs)
    })
}

#[test]
fn test_eq_3() {
    with_default_globals(|| {
        let test_res = string_to_ts("");
        let test_eqs = string_to_ts("");
        assert_eq!(test_res, test_eqs)
    })
}

#[test]
fn test_diseq_0() {
    with_default_globals(|| {
        let test_res = string_to_ts("::bar::baz");
        let test_eqs = string_to_ts("bar::baz");
        assert_eq!(test_res == test_eqs, false)
    })
}

#[test]
fn test_diseq_1() {
    with_default_globals(|| {
        let test_res = string_to_ts("(bar,baz)");
        let test_eqs = string_to_ts("bar,baz");
        assert_eq!(test_res == test_eqs, false)
    })
}

#[test]
fn test_is_empty() {
    with_default_globals(|| {
        let test0: TokenStream = Vec::<TokenTree>::new().into_iter().collect();
        let test1: TokenStream =
            TokenTree::token(token::Ident(Name::intern("a"), false), sp(0, 1)).into();
        let test2 = string_to_ts("foo(bar::baz)");

        assert_eq!(test0.is_empty(), true);
        assert_eq!(test1.is_empty(), false);
        assert_eq!(test2.is_empty(), false);
    })
}

#[test]
fn test_dotdotdot() {
    with_default_globals(|| {
        let mut builder = TokenStreamBuilder::new();
        builder.push(TokenTree::token(token::Dot, sp(0, 1)).joint());
        builder.push(TokenTree::token(token::Dot, sp(1, 2)).joint());
        builder.push(TokenTree::token(token::Dot, sp(2, 3)));
        let stream = builder.build();
        assert!(stream.eq_unspanned(&string_to_ts("...")));
        assert_eq!(stream.trees().count(), 1);
    })
}
