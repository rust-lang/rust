use std::pattern::*;

// This macro makes it easier to write
// tests that do a series of iterations
macro_rules! search_asserts {
    ($haystack:expr, $needle:expr, $testname:literal, $($func:ident => $result:expr),*) => {
        let mut searcher = $needle.into_searcher($haystack);
        let arr = [$( searcher.$func().into_step(stringify!($func)) ),*];
        assert_eq!(&arr[..], &[$($result),*], $testname);
    }
}

/// Combined enum for the results of next() and next_match()/next_reject()
#[derive(Debug, PartialEq, Eq)]
enum Step {
    // variant names purposely chosen to
    // be the same length for easy alignment
    Matches(usize, usize),
    Rejects(usize, usize),
    Done,
}

use Step::*;

trait IntoStep {
    fn into_step(self, method_name: &str) -> Step;
}

impl IntoStep for SearchStep {
    fn into_step(self, _name: &str) -> Step {
        match self {
            SearchStep::Match(s, e) => Matches(s, e),
            SearchStep::Reject(s, e) => Rejects(s, e),
            SearchStep::Done => Done,
        }
    }
}

impl IntoStep for Option<(usize, usize)> {
    fn into_step(self, method_name: &str) -> Step {
        let is_reject = method_name.starts_with("next_reject");
        match self {
            Some((s, e)) if is_reject => Rejects(s, e),
            Some((s, e)) => Matches(s, e),
            None => Done,
        }
    }
}

// FIXME(Manishearth) these tests focus on single-character searching  (CharSearcher)
// and on next()/next_match(), not next_reject(). This is because
// the memchr changes make next_match() for single chars complex, but next_reject()
// continues to use next() under the hood. We should add more test cases for all
// of these, as well as tests for StrSearcher and higher level tests for str::find() (etc)

#[test]
fn test_simple_iteration() {
    search_asserts!(
        "abcdeabcd",
        'a',
        "forward iteration for ASCII string",
        next => Matches(0, 1),
        next => Rejects(1, 5),
        next => Matches(5, 6),
        next => Rejects(6, 9),
        next => Done
    );

    search_asserts!(
        "abcdeabcd",
        'a',
        "reverse iteration for ASCII string",
        next_back => Rejects(6, 9),
        next_back => Matches(5, 6),
        next_back => Rejects(1, 5),
        next_back => Matches(0, 1),
        next_back => Done
    );

    search_asserts!(
        "我爱我的猫",
        '我',
        "forward iteration for Chinese string",
        next => Matches(0, 3),
        next => Rejects(3, 6),
        next => Matches(6, 9),
        next => Rejects(9, 15),
        next => Done
    );

    search_asserts!(
        "我的猫说meow",
        'm',
        "forward iteration for mixed string",
        next => Rejects(0, 12),
        next => Matches(12, 13),
        next => Rejects(13, 16),
        next => Done
    );

    search_asserts!(
        "我的猫说meow",
        '猫',
        "reverse iteration for mixed string",
        next_back => Rejects(9, 16),
        next_back => Matches(6, 9),
        next_back => Rejects(0, 6),
        next_back => Done
    );
}

#[test]
fn test_simple_search() {
    search_asserts!(
        "abcdeabcdeabcde",
        'a',
        "next_match for ASCII string",
        next_match => Matches(0, 1),
        next_match => Matches(5, 6),
        next_match => Matches(10, 11),
        next_match => Done
    );

    search_asserts!(
        "abcdeabcdeabcde",
        'a',
        "next_match_back for ASCII string",
        next_match_back => Matches(10, 11),
        next_match_back => Matches(5, 6),
        next_match_back => Matches(0, 1),
        next_match_back => Done
    );

    search_asserts!(
        "abcdeab",
        'a',
        "next_reject for ASCII string",
        next_reject => Rejects(1, 2),
        next_reject => Rejects(2, 3),
        next_match => Matches(5, 6),
        next_reject => Rejects(6, 7),
        next_reject => Done
    );

    search_asserts!(
        "abcdeabcdeabcde",
        'a',
        "next_reject_back for ASCII string",
        next_reject_back => Rejects(14, 15),
        next_reject_back => Rejects(13, 14),
        next_match_back  => Matches(10, 11),
        next_reject_back => Rejects(9, 10),
        next_reject_back => Rejects(8, 9),
        next_reject_back => Rejects(7, 8)
    );
}

// Á, 각, ก, 😀 all end in 0x81
// 🁀, ᘀ do not end in 0x81 but contain the byte
// ꁁ has 0x81 as its second and third bytes.
//
// The memchr-using implementation of next_match
// and next_match_back temporarily violate
// the property that the search is always on a unicode boundary,
// which is fine as long as this never reaches next() or next_back().
// So we test if next() is correct after each next_match() as well.
const STRESS: &str = "Áa🁀bÁꁁfg😁각กᘀ각aÁ각ꁁก😁a";

#[test]
fn test_stress_indices() {
    // this isn't really a test, more of documentation on the indices of each character in the stresstest string
    search_asserts!(
        STRESS,
        |_| true,
        "Indices of characters in stress test",
        next => Matches(0, 2),   // Á
        next => Matches(2, 3),   // a
        next => Matches(3, 7),   // 🁀
        next => Matches(7, 8),   // b
        next => Matches(8, 10),  // Á
        next => Matches(10, 13), // ꁁ
        next => Matches(13, 14), // f
        next => Matches(14, 15), // g
        next => Matches(15, 19), // 😀
        next => Matches(19, 22), // 각
        next => Matches(22, 25), // ก
        next => Matches(25, 28), // ᘀ
        next => Matches(28, 31), // 각
        next => Matches(31, 32), // a
        next => Matches(32, 34), // Á
        next => Matches(34, 37), // 각
        next => Matches(37, 40), // ꁁ
        next => Matches(40, 43), // ก
        next => Matches(43, 47), // 😀
        next => Matches(47, 48), // a
        next => Done
    );
}

#[test]
fn test_forward_search_shared_bytes() {
    search_asserts!(
        STRESS,
        'Á',
        "Forward search for two-byte Latin character",
        next_match => Matches(0, 2),
        next_match => Matches(8, 10),
        next_match => Matches(32, 34),
        next_match => Done
    );

    search_asserts!(
        STRESS,
        'Á',
        "Forward search for two-byte Latin character; check if next() still works",
        next_match => Matches(0, 2),
        next       => Rejects(2, 8),
        next_match => Matches(8, 10),
        next       => Rejects(10, 32),
        next_match => Matches(32, 34),
        next       => Rejects(34, 48),
        next_match => Done
    );

    search_asserts!(
        STRESS,
        '각',
        "Forward search for three-byte Hangul character",
        next_match => Matches(19, 22),
        next       => Rejects(22, 28),
        next_match => Matches(28, 31),
        next_match => Matches(34, 37),
        next_match => Done
    );

    search_asserts!(
        STRESS,
        '각',
        "Forward search for three-byte Hangul character; check if next() still works",
        next_match => Matches(19, 22),
        next       => Rejects(22, 28),
        next_match => Matches(28, 31),
        next       => Rejects(31, 34),
        next_match => Matches(34, 37),
        next       => Rejects(37, 48),
        next_match => Done
    );

    search_asserts!(
        STRESS,
        'ก',
        "Forward search for three-byte Thai character",
        next_match => Matches(22, 25),
        next       => Rejects(25, 40),
        next_match => Matches(40, 43),
        next       => Rejects(43, 48),
        next_match => Done
    );

    search_asserts!(
        STRESS,
        'ก',
        "Forward search for three-byte Thai character; check if next() still works",
        next_match => Matches(22, 25),
        next       => Rejects(25, 40),
        next_match => Matches(40, 43),
        next       => Rejects(43, 48),
        next_match => Done
    );

    search_asserts!(
        STRESS,
        '😁',
        "Forward search for four-byte emoji",
        next_match => Matches(15, 19),
        next       => Rejects(19, 43),
        next_match => Matches(43, 47),
        next       => Rejects(47, 48),
        next_match => Done
    );

    search_asserts!(
        STRESS,
        '😁',
        "Forward search for four-byte emoji; check if next() still works",
        next_match => Matches(15, 19),
        next       => Rejects(19, 43),
        next_match => Matches(43, 47),
        next       => Rejects(47, 48),
        next_match => Done
    );

    search_asserts!(
        STRESS,
        'ꁁ',
        "Forward search for three-byte Yi character with repeated bytes",
        next_match => Matches(10, 13),
        next       => Rejects(13, 37),
        next_match => Matches(37, 40),
        next       => Rejects(40, 48),
        next_match => Done
    );

    search_asserts!(
        STRESS,
        'ꁁ',
        "Forward search for three-byte Yi character with repeated bytes; check if next() still works",
        next_match => Matches(10, 13),
        next       => Rejects(13, 37),
        next_match => Matches(37, 40),
        next       => Rejects(40, 48),
        next_match => Done
    );
}

#[test]
fn test_reverse_search_shared_bytes() {
    search_asserts!(
        STRESS,
        'Á',
        "Reverse search for two-byte Latin character",
        next_match_back => Matches(32, 34),
        next_match_back => Matches(8, 10),
        next_match_back => Matches(0, 2),
        next_match_back => Done
    );

    search_asserts!(
        STRESS,
        'Á',
        "Reverse search for two-byte Latin character; check if next_back() still works",
        next_match_back => Matches(32, 34),
        next_back       => Rejects(10, 32),
        next_match_back => Matches(8, 10),
        next_back       => Rejects(2, 8),
        next_match_back => Matches(0, 2),
        next_back       => Done
    );

    search_asserts!(
        STRESS,
        '각',
        "Reverse search for three-byte Hangul character",
        next_match_back => Matches(34, 37),
        next_back       => Rejects(31, 34),
        next_match_back => Matches(28, 31),
        next_match_back => Matches(19, 22),
        next_match_back => Done
    );

    search_asserts!(
        STRESS,
        '각',
        "Reverse search for three-byte Hangul character; check if next_back() still works",
        next_match_back => Matches(34, 37),
        next_back       => Rejects(31, 34),
        next_match_back => Matches(28, 31),
        next_back       => Rejects(22, 28),
        next_match_back => Matches(19, 22),
        next_back       => Rejects(0, 19),
        next_match_back => Done
    );

    search_asserts!(
        STRESS,
        'ก',
        "Reverse search for three-byte Thai character",
        next_match_back => Matches(40, 43),
        next_back       => Rejects(25, 40),
        next_match_back => Matches(22, 25),
        next_back       => Rejects(0, 22),
        next_match_back => Done
    );

    search_asserts!(
        STRESS,
        'ก',
        "Reverse search for three-byte Thai character; check if next_back() still works",
        next_match_back => Matches(40, 43),
        next_back       => Rejects(25, 40),
        next_match_back => Matches(22, 25),
        next_back       => Rejects(0, 22),
        next_match_back => Done
    );

    search_asserts!(
        STRESS,
        '😁',
        "Reverse search for four-byte emoji",
        next_match_back => Matches(43, 47),
        next_back       => Rejects(19, 43),
        next_match_back => Matches(15, 19),
        next_back       => Rejects(0, 15),
        next_match_back => Done
    );

    search_asserts!(
        STRESS,
        '😁',
        "Reverse search for four-byte emoji; check if next_back() still works",
        next_match_back => Matches(43, 47),
        next_back       => Rejects(19, 43),
        next_match_back => Matches(15, 19),
        next_back       => Rejects(0, 15),
        next_match_back => Done
    );

    search_asserts!(
        STRESS,
        'ꁁ',
        "Reverse search for three-byte Yi character with repeated bytes",
        next_match_back => Matches(37, 40),
        next_back       => Rejects(13, 37),
        next_match_back => Matches(10, 13),
        next_back       => Rejects(0, 10),
        next_match_back => Done
    );

    search_asserts!(
        STRESS,
        'ꁁ',
        "Reverse search for three-byte Yi character with repeated bytes; check if next_back() still works",
        next_match_back => Matches(37, 40),
        next_back       => Rejects(13, 37),
        next_match_back => Matches(10, 13),
        next_back       => Rejects(0, 10),
        next_match_back => Done
    );
}

#[test]
fn double_ended_regression_test() {
    // https://github.com/rust-lang/rust/issues/47175
    // Ensures that double ended searching comes to a convergence
    search_asserts!(
        "abcdeabcdeabcde",
        'a',
        "alternating double ended search",
        next_match      => Matches(0, 1),
        next_match_back => Matches(10, 11),
        next_match      => Matches(5, 6),
        next_match_back => Done
    );
    search_asserts!(
        "abcdeabcdeabcde",
        'a',
        "triple double ended search for a",
        next_match      => Matches(0, 1),
        next_match_back => Matches(10, 11),
        next_match_back => Matches(5, 6),
        next_match_back => Done
    );
    search_asserts!(
        "abcdeabcdeabcde",
        'd',
        "triple double ended search for d",
        next_match      => Matches(3, 4),
        next_match_back => Matches(13, 14),
        next_match_back => Matches(8, 9),
        next_match_back => Done
    );
    search_asserts!(
        STRESS,
        'Á',
        "Double ended search for two-byte Latin character",
        next_match      => Matches(0, 2),
        next_match_back => Matches(32, 34),
        next_match      => Matches(8, 10),
        next_match_back => Done
    );
    search_asserts!(
        STRESS,
        '각',
        "Reverse double ended search for three-byte Hangul character",
        next_match_back => Matches(34, 37),
        next_back       => Rejects(31, 34),
        next_match      => Matches(19, 22),
        next            => Rejects(22, 28),
        next_match_back => Matches(28, 31),
        next_match      => Done
    );
    search_asserts!(
        STRESS,
        'ก',
        "Double ended search for three-byte Thai character",
        next_match      => Matches(22, 25),
        next_back       => Rejects(43, 48),
        next            => Rejects(25, 40),
        next_match_back => Matches(40, 43),
        next_match      => Done
    );
    search_asserts!(
        STRESS,
        '😁',
        "Double ended search for four-byte emoji",
        next_match_back => Matches(43, 47),
        next            => Rejects(0, 15),
        next_match      => Matches(15, 19),
        next_back       => Rejects(19, 43),
        next_match      => Done
    );
    search_asserts!(
        STRESS,
        'ꁁ',
        "Double ended search for three-byte Yi character with repeated bytes",
        next_match      => Matches(10, 13),
        next            => Rejects(13, 37),
        next_match_back => Matches(37, 40),
        next_back       => Done
    );
}
