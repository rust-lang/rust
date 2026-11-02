//@ run-pass
//@ edition: 2021
//! Executable form of the safety contract on `Pattern`: the searcher returned by
//! `into_searcher` must search exactly the haystack it was given (same pointer and length).
//!
//! `str::split_once`, `trim_matches`, `replace`, `split`, ... slice the string they were
//! called on using indices reported by the searcher, with no further checks.
//!
//! Part 1 checks the contract for every `Pattern` impl in the standard library.
//! Part 2 checks that a conforming user-written `unsafe impl` works through every method
//! that relies on the contract, by comparing against the wrapped pattern.

#![feature(pattern, string_remove_matches)]

use std::str::pattern::{DoubleEndedSearcher, Pattern, ReverseSearcher, Searcher};

/// Includes empty, ASCII, multi-byte, and sub-slices of a larger string (so the
/// haystack pointer is not the start of its allocation).
fn haystacks() -> Vec<&'static str> {
    const BIG: &str = "xxbananaxx\u{e9}a\u{65e5}\u{672c}a\u{8a9e}a";
    vec![
        "",
        "a",
        "aaa",
        "banana",
        "abcabc",
        "xaxbx",
        "\u{e9}",
        "a\u{e9}a",
        "\u{65e5}\u{672c}a\u{8a9e}a",
        &BIG[2..8],
        &BIG[0..0],
        &BIG[BIG.len() - 4..],
        BIG,
    ]
}

// ---------------------------------------------------------------------------
// Part 1: every in-tree pattern upholds the contract.

fn assert_contract<P: Pattern>(name: &str, p: P, h: &str) {
    let s = p.into_searcher(h);
    let sh = s.haystack();
    assert!(
        sh.as_ptr() == h.as_ptr() && sh.len() == h.len(),
        "{name}: searcher haystack differs from the one passed in ({h:?})"
    );
    // `haystack()` is documented to always return the same string.
    let sh2 = s.haystack();
    assert!(sh.as_ptr() == sh2.as_ptr() && sh.len() == sh2.len(), "{name}: haystack() is unstable");
}

fn contract_for_all_in_tree_patterns() {
    let string = String::from("an");
    for h in haystacks() {
        assert_contract("char", 'a', h);
        assert_contract("&str", "an", h);
        assert_contract("&&str", &"an", h);
        assert_contract("&String", &string, h);
        assert_contract("[char; 2]", ['a', 'n'], h);
        assert_contract("&[char; 2]", &['a', 'n'], h);
        assert_contract("&[char]", &['a', 'n'][..], h);
        assert_contract("FnMut(char) -> bool", |c: char| c == 'a', h);
        assert_contract("fn(char) -> bool", char::is_alphabetic as fn(char) -> bool, h);
    }
}

// ---------------------------------------------------------------------------
// Part 2: a conforming wrapper behaves exactly like the pattern it wraps.

struct Fwd<P>(P);

// SAFETY: forwards `haystack` unchanged and returns the wrapped pattern's searcher as is.
unsafe impl<P: Pattern> Pattern for Fwd<P> {
    type Searcher<'a> = P::Searcher<'a>;

    fn into_searcher<'a>(self, haystack: &'a str) -> P::Searcher<'a> {
        self.0.into_searcher(haystack)
    }
}

macro_rules! same {
    ($h:expr, $p:expr, |$q:ident| $e:expr) => {{
        let plain = { let $q = $p.clone(); $e };
        let fwd = { let $q = Fwd($p.clone()); $e };
        assert_eq!(plain, fwd, "{} on {:?}", stringify!($e), $h);
    }};
}

fn same_as_wrapped<P>(p: P)
where
    P: Pattern + Clone,
    for<'a> P::Searcher<'a>: ReverseSearcher<'a>,
{
    for h in haystacks() {
        same!(h, p, |q| h.split_once(q));
        same!(h, p, |q| h.rsplit_once(q));
        same!(h, p, |q| h.split(q).collect::<Vec<_>>());
        same!(h, p, |q| h.rsplit(q).collect::<Vec<_>>());
        same!(h, p, |q| h.splitn(2, q).collect::<Vec<_>>());
        same!(h, p, |q| h.rsplitn(2, q).collect::<Vec<_>>());
        same!(h, p, |q| h.split_inclusive(q).collect::<Vec<_>>());
        same!(h, p, |q| h.split_terminator(q).collect::<Vec<_>>());
        same!(h, p, |q| h.rsplit_terminator(q).collect::<Vec<_>>());
        same!(h, p, |q| h.matches(q).collect::<Vec<_>>());
        same!(h, p, |q| h.rmatches(q).collect::<Vec<_>>());
        same!(h, p, |q| h.match_indices(q).collect::<Vec<_>>());
        same!(h, p, |q| h.rmatch_indices(q).collect::<Vec<_>>());
        same!(h, p, |q| h.find(q));
        same!(h, p, |q| h.rfind(q));
        same!(h, p, |q| h.contains(q));
        same!(h, p, |q| h.starts_with(q));
        same!(h, p, |q| h.ends_with(q));
        same!(h, p, |q| h.strip_prefix(q));
        same!(h, p, |q| h.strip_suffix(q));
        same!(h, p, |q| h.trim_start_matches(q));
        same!(h, p, |q| h.trim_end_matches(q));
        same!(h, p, |q| h.replace(q, "<>"));
        same!(h, p, |q| h.replacen(q, "<>", 1));
        same!(h, p, |q| {
            let mut s = h.to_string();
            s.remove_matches(q);
            s
        });
    }
}

fn same_as_wrapped_double_ended<P>(p: P)
where
    P: Pattern + Clone,
    for<'a> P::Searcher<'a>: DoubleEndedSearcher<'a>,
{
    for h in haystacks() {
        same!(h, p, |q| h.trim_matches(q));
    }
}

fn wrapper_matches_wrapped_pattern() {
    same_as_wrapped('a');
    same_as_wrapped("an");
    same_as_wrapped("");
    same_as_wrapped(['a', 'n']);
    same_as_wrapped(&['a', 'n'][..]);
    same_as_wrapped(|c: char| c == 'a' || c == '\u{e9}');

    same_as_wrapped_double_ended('a');
    same_as_wrapped_double_ended(['a', 'n']);
    same_as_wrapped_double_ended(&['a', 'n'][..]);
    same_as_wrapped_double_ended(|c: char| c == 'a' || c == '\u{e9}');
}

fn main() {
    contract_for_all_in_tree_patterns();
    wrapper_matches_wrapped_pattern();
}
