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

    // this test should generate something that will run memchr
    search_asserts!(
        STRESS,
        'x',
        "Indices of characters in stress test",
        next => Rejects(0, 48), // no character matches
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

#[test]
fn test_try_next_code_point_fwd_and_rev() {
    use core::str::{try_next_code_point, try_next_code_point_reverse};

    #[track_caller]
    fn check_fwbw(input: &[u8], expected: Option<(char, usize)>) {
        let mut store = Vec::with_capacity(input.len() + 1);

        assert_eq!(try_next_code_point(input), expected, "fw single");
        assert_eq!(try_next_code_point_reverse(input), expected, "bw single");

        if input.is_empty() {
            return;
        }

        store.extend_from_slice(input);
        store.push(b'a');
        assert_eq!(try_next_code_point(&store), expected, "fw ext");

        store.clear();
        store.push(b'a');
        store.extend_from_slice(input);
        assert_eq!(try_next_code_point_reverse(&store), expected, "bw, ext");
    }

    // edge case
    check_fwbw(&[], None); // edge case

    // valid cases
    check_fwbw(&[0x41], Some(('A', 1))); // 1 byte
    check_fwbw(&[0xc3, 0xa9], Some(('\u{00e9}', 2))); // 2 byte
    check_fwbw(&[0xe2, 0x82, 0xac], Some(('\u{20ac}', 3))); // 3 byte
    check_fwbw(&[0xf0, 0x9f, 0x98, 0x8a], Some(('\u{1f60a}', 4))); // 4 byte

    // overlong encoding
    check_fwbw(&[0xc0, 0xaf], None); // '\u{002f}' AKA '/'
    check_fwbw(&[0xc1, 0x81], None); // '\u{0041}' AKA 'A'
    check_fwbw(&[0xe0, 0x81, 0x81], None); // '\u{0041}' AKA 'A'
    check_fwbw(&[0xe0, 0x83, 0xa9], None); // '\u{00e9}' AKA 'é', should be 2 bytes

    // surrogates
    check_fwbw(&[0xed, 0xa0, 0x80], None);
    check_fwbw(&[0xed, 0xb0, 0x80], None);
    check_fwbw(&[0xed, 0xa0, 0xbd, 0xed, 0xb8, 0x8a], None);

    // unattached continuation
    check_fwbw(&[0x80], None);

    // truncated
    check_fwbw(&[0xe2, 0x82], None);

    // out of range
    check_fwbw(&[0xf5, 0x80, 0x80, 0x80], None);
}

#[test]
fn two_way_next_reject_skips_matches() {
    // `next_reject` must not report the matched regions as rejects

    // plen - pattern len
    // ulen - unmatch len
    // haystack is always a concatenation of [Match, Reject, Match]
    #[track_caller]
    fn check_fw_bw<H, T>(haystack: H, pat: T, plen: usize, ulen: usize)
    where
        H: Haystack,
        T: Pattern<H> + Copy,
        T::Searcher: ReverseSearcher<H>,
    {
        // fragments
        let f1 = (0, plen);
        let f2 = (f1.1, f1.1 + ulen);
        let f3 = (f2.1, f2.1 + plen);

        let mut searcher = pat.into_searcher(haystack);
        let (start, end) = searcher.next_reject().expect("fw reject");
        assert_eq!(start, f2.0, "fw reject start");
        assert!(start < end && end <= f2.1, "fw reject must be a non-empty part of f2");
        assert_eq!(searcher.next_match(), Some(f3), "fw match");
        assert_eq!(searcher.next_match(), None, "fw end");

        let mut searcher = pat.into_searcher(haystack);
        let (start, end) = searcher.next_reject_back().expect("bw reject");
        assert_eq!(end, f2.1, "bw reject end");
        assert!(f2.0 <= start && start < end, "bw reject must be a non-empty part of f2");
        assert_eq!(searcher.next_match_back(), Some(f1), "bw match");
        assert_eq!(searcher.next_match_back(), None, "bw end");
    }
    check_fw_bw("XYZabcXYZ", "XYZ", 3, 3);

    // single char 1..4 byte str
    check_fw_bw("XabcX", "X", 1, 3);
    check_fw_bw("\u{00e9}abc\u{00e9}", "\u{00e9}", 2, 3);
    check_fw_bw("\u{20ac}abc\u{20ac}", "\u{20ac}", 3, 3);
    check_fw_bw("\u{1f60a}abc\u{1f60a}", "\u{1f60a}", 4, 3);

    // single char 2..4 byte
    check_fw_bw("\u{00e9}abc\u{00e9}", '\u{00e9}', 2, 3);
    check_fw_bw("\u{20ac}abc\u{20ac}", '\u{20ac}', 3, 3);
    check_fw_bw("\u{1f60a}abc\u{1f60a}", '\u{1f60a}', 4, 3);

    check_fw_bw("\u{00e9}abc\u{00e9}", ['\u{00e9}'], 2, 3);
    check_fw_bw("\u{20ac}abc\u{20ac}", ['\u{20ac}'], 3, 3);
    check_fw_bw("\u{1f60a}abc\u{1f60a}", ['\u{1f60a}'], 4, 3);

    check_fw_bw("\u{00e9}abc\u{00e9}", |c| c == '\u{00e9}', 2, 3);
    check_fw_bw("\u{20ac}abc\u{20ac}", |c| c == '\u{20ac}', 3, 3);
    check_fw_bw("\u{1f60a}abc\u{1f60a}", |c| c == '\u{1f60a}', 4, 3);
}

#[test]
fn two_way_next_reject_never_reports_empty_reject() {
    // A reject must never be an empty range

    // Haystack fully covered by the pattern: there are no rejects at all.
    let mut searcher = "XYZ".into_searcher("XYZ");
    assert_eq!(searcher.next_reject(), None);
    assert_eq!(searcher.next_reject(), None);

    let mut searcher = "XYZ".into_searcher("XYZ");
    assert_eq!(searcher.next_reject_back(), None);
    assert_eq!(searcher.next_reject_back(), None);

    // Haystack ending with a match: the last reject is followed by `None`,
    // not by an empty reject after the final match.
    let mut searcher = "XYZ".into_searcher("XYZabcXYZ");
    assert_eq!(searcher.next_reject(), Some((3, 6)));
    assert_eq!(searcher.next_reject(), None);

    let mut searcher = "XYZ".into_searcher("XYZabcXYZ");
    assert_eq!(searcher.next_reject_back(), Some((3, 6)));
    assert_eq!(searcher.next_reject_back(), None);
}

#[test]
fn str_searcher_reject_back_preserves_start_on_bad_input() {
    let haystack = core::str_bytes::Bytes::from_bytes(&[0x80, b'x', b'y', b'z'][..]);
    let mut searcher = "xyz".into_searcher(haystack);
    assert_eq!(searcher.next_reject_back(), Some((0, 1)));
    assert_eq!(searcher.next_reject_back(), None);
}
