use std::str::pattern::*;

// This macro makes it easier to write
// tests that do a series of iterations
macro_rules! search_asserts {
    ($haystack:expr, $needle:expr, $testname:expr, [$($func:ident),*], $result:expr) => {
        let mut searcher = $needle.into_searcher($haystack);
        let arr = [$( Step::from(searcher.$func()) ),*];
        assert_eq!(&arr[..], &$result, $testname);
    }
}

/// Combined enum for the results of next() and next_match()/next_reject()
#[derive(Debug, PartialEq, Eq)]
enum Step {
    // variant names purposely chosen to
    // be the same length for easy alignment
    Matches(usize, usize),
    Rejects(usize, usize),
    InRange(usize, usize),
    Done,
}

use self::Step::*;

impl From<SearchStep> for Step {
    fn from(x: SearchStep) -> Self {
        match x {
            SearchStep::Match(a, b) => Matches(a, b),
            SearchStep::Reject(a, b) => Rejects(a, b),
            SearchStep::Done => Done,
        }
    }
}

impl From<Option<(usize, usize)>> for Step {
    fn from(x: Option<(usize, usize)>) -> Self {
        match x {
            Some((a, b)) => InRange(a, b),
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
        // a            b              c              d              e              a              b              c              d              EOF
        [next, next, next, next, next, next, next, next, next, next],
        [
            Matches(0, 1),
            Rejects(1, 2),
            Rejects(2, 3),
            Rejects(3, 4),
            Rejects(4, 5),
            Matches(5, 6),
            Rejects(6, 7),
            Rejects(7, 8),
            Rejects(8, 9),
            Done
        ]
    );

    search_asserts!(
        "abcdeabcd",
        'a',
        "reverse iteration for ASCII string",
        // d            c              b              a            e                d              c              b              a             EOF
        [
            next_back, next_back, next_back, next_back, next_back, next_back, next_back, next_back,
            next_back, next_back
        ],
        [
            Rejects(8, 9),
            Rejects(7, 8),
            Rejects(6, 7),
            Matches(5, 6),
            Rejects(4, 5),
            Rejects(3, 4),
            Rejects(2, 3),
            Rejects(1, 2),
            Matches(0, 1),
            Done
        ]
    );

    search_asserts!(
        "我爱我的猫",
        '我',
        "forward iteration for Chinese string",
        // 我           愛             我             的              貓               EOF
        [next, next, next, next, next, next],
        [Matches(0, 3), Rejects(3, 6), Matches(6, 9), Rejects(9, 12), Rejects(12, 15), Done]
    );

    search_asserts!(
        "我的猫说meow",
        'm',
        "forward iteration for mixed string",
        // 我           的             猫             说              m                e                o                w                EOF
        [next, next, next, next, next, next, next, next, next],
        [
            Rejects(0, 3),
            Rejects(3, 6),
            Rejects(6, 9),
            Rejects(9, 12),
            Matches(12, 13),
            Rejects(13, 14),
            Rejects(14, 15),
            Rejects(15, 16),
            Done
        ]
    );

    search_asserts!(
        "我的猫说meow",
        '猫',
        "reverse iteration for mixed string",
        // w             o                 e                m                说              猫             的             我             EOF
        [
            next_back, next_back, next_back, next_back, next_back, next_back, next_back, next_back,
            next_back
        ],
        [
            Rejects(15, 16),
            Rejects(14, 15),
            Rejects(13, 14),
            Rejects(12, 13),
            Rejects(9, 12),
            Matches(6, 9),
            Rejects(3, 6),
            Rejects(0, 3),
            Done
        ]
    );
}

#[test]
fn test_simple_search() {
    search_asserts!(
        "abcdeabcdeabcde",
        'a',
        "next_match for ASCII string",
        [next_match, next_match, next_match, next_match],
        [InRange(0, 1), InRange(5, 6), InRange(10, 11), Done]
    );

    search_asserts!(
        "abcdeabcdeabcde",
        'a',
        "next_match_back for ASCII string",
        [next_match_back, next_match_back, next_match_back, next_match_back],
        [InRange(10, 11), InRange(5, 6), InRange(0, 1), Done]
    );

    search_asserts!(
        "abcdeab",
        'a',
        "next_reject for ASCII string",
        [next_reject, next_reject, next_match, next_reject, next_reject],
        [InRange(1, 2), InRange(2, 3), InRange(5, 6), InRange(6, 7), Done]
    );

    search_asserts!(
        "abcdeabcdeabcde",
        'a',
        "next_reject_back for ASCII string",
        [
            next_reject_back,
            next_reject_back,
            next_match_back,
            next_reject_back,
            next_reject_back,
            next_reject_back
        ],
        [
            InRange(14, 15),
            InRange(13, 14),
            InRange(10, 11),
            InRange(9, 10),
            InRange(8, 9),
            InRange(7, 8)
        ]
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
        'x',
        "Indices of characters in stress test",
        [
            next, next, next, next, next, next, next, next, next, next, next, next, next, next,
            next, next, next, next, next, next, next
        ],
        [
            Rejects(0, 2),   // Á
            Rejects(2, 3),   // a
            Rejects(3, 7),   // 🁀
            Rejects(7, 8),   // b
            Rejects(8, 10),  // Á
            Rejects(10, 13), // ꁁ
            Rejects(13, 14), // f
            Rejects(14, 15), // g
            Rejects(15, 19), // 😀
            Rejects(19, 22), // 각
            Rejects(22, 25), // ก
            Rejects(25, 28), // ᘀ
            Rejects(28, 31), // 각
            Rejects(31, 32), // a
            Rejects(32, 34), // Á
            Rejects(34, 37), // 각
            Rejects(37, 40), // ꁁ
            Rejects(40, 43), // ก
            Rejects(43, 47), // 😀
            Rejects(47, 48), // a
            Done
        ]
    );
}

#[test]
fn test_forward_search_shared_bytes() {
    search_asserts!(
        STRESS,
        'Á',
        "Forward search for two-byte Latin character",
        [next_match, next_match, next_match, next_match],
        [InRange(0, 2), InRange(8, 10), InRange(32, 34), Done]
    );

    search_asserts!(
        STRESS,
        'Á',
        "Forward search for two-byte Latin character; check if next() still works",
        [next_match, next, next_match, next, next_match, next, next_match],
        [
            InRange(0, 2),
            Rejects(2, 3),
            InRange(8, 10),
            Rejects(10, 13),
            InRange(32, 34),
            Rejects(34, 37),
            Done
        ]
    );

    search_asserts!(
        STRESS,
        '각',
        "Forward search for three-byte Hangul character",
        [next_match, next, next_match, next_match, next_match],
        [InRange(19, 22), Rejects(22, 25), InRange(28, 31), InRange(34, 37), Done]
    );

    search_asserts!(
        STRESS,
        '각',
        "Forward search for three-byte Hangul character; check if next() still works",
        [next_match, next, next_match, next, next_match, next, next_match],
        [
            InRange(19, 22),
            Rejects(22, 25),
            InRange(28, 31),
            Rejects(31, 32),
            InRange(34, 37),
            Rejects(37, 40),
            Done
        ]
    );

    search_asserts!(
        STRESS,
        'ก',
        "Forward search for three-byte Thai character",
        [next_match, next, next_match, next, next_match],
        [InRange(22, 25), Rejects(25, 28), InRange(40, 43), Rejects(43, 47), Done]
    );

    search_asserts!(
        STRESS,
        'ก',
        "Forward search for three-byte Thai character; check if next() still works",
        [next_match, next, next_match, next, next_match],
        [InRange(22, 25), Rejects(25, 28), InRange(40, 43), Rejects(43, 47), Done]
    );

    search_asserts!(
        STRESS,
        '😁',
        "Forward search for four-byte emoji",
        [next_match, next, next_match, next, next_match],
        [InRange(15, 19), Rejects(19, 22), InRange(43, 47), Rejects(47, 48), Done]
    );

    search_asserts!(
        STRESS,
        '😁',
        "Forward search for four-byte emoji; check if next() still works",
        [next_match, next, next_match, next, next_match],
        [InRange(15, 19), Rejects(19, 22), InRange(43, 47), Rejects(47, 48), Done]
    );

    search_asserts!(
        STRESS,
        'ꁁ',
        "Forward search for three-byte Yi character with repeated bytes",
        [next_match, next, next_match, next, next_match],
        [InRange(10, 13), Rejects(13, 14), InRange(37, 40), Rejects(40, 43), Done]
    );

    search_asserts!(
        STRESS,
        'ꁁ',
        "Forward search for three-byte Yi character with repeated bytes; check if next() still works",
        [next_match, next, next_match, next, next_match],
        [InRange(10, 13), Rejects(13, 14), InRange(37, 40), Rejects(40, 43), Done]
    );
}

#[test]
fn test_reverse_search_shared_bytes() {
    search_asserts!(
        STRESS,
        'Á',
        "Reverse search for two-byte Latin character",
        [next_match_back, next_match_back, next_match_back, next_match_back],
        [InRange(32, 34), InRange(8, 10), InRange(0, 2), Done]
    );

    search_asserts!(
        STRESS,
        'Á',
        "Reverse search for two-byte Latin character; check if next_back() still works",
        [next_match_back, next_back, next_match_back, next_back, next_match_back, next_back],
        [InRange(32, 34), Rejects(31, 32), InRange(8, 10), Rejects(7, 8), InRange(0, 2), Done]
    );

    search_asserts!(
        STRESS,
        '각',
        "Reverse search for three-byte Hangul character",
        [next_match_back, next_back, next_match_back, next_match_back, next_match_back],
        [InRange(34, 37), Rejects(32, 34), InRange(28, 31), InRange(19, 22), Done]
    );

    search_asserts!(
        STRESS,
        '각',
        "Reverse search for three-byte Hangul character; check if next_back() still works",
        [
            next_match_back,
            next_back,
            next_match_back,
            next_back,
            next_match_back,
            next_back,
            next_match_back
        ],
        [
            InRange(34, 37),
            Rejects(32, 34),
            InRange(28, 31),
            Rejects(25, 28),
            InRange(19, 22),
            Rejects(15, 19),
            Done
        ]
    );

    search_asserts!(
        STRESS,
        'ก',
        "Reverse search for three-byte Thai character",
        [next_match_back, next_back, next_match_back, next_back, next_match_back],
        [InRange(40, 43), Rejects(37, 40), InRange(22, 25), Rejects(19, 22), Done]
    );

    search_asserts!(
        STRESS,
        'ก',
        "Reverse search for three-byte Thai character; check if next_back() still works",
        [next_match_back, next_back, next_match_back, next_back, next_match_back],
        [InRange(40, 43), Rejects(37, 40), InRange(22, 25), Rejects(19, 22), Done]
    );

    search_asserts!(
        STRESS,
        '😁',
        "Reverse search for four-byte emoji",
        [next_match_back, next_back, next_match_back, next_back, next_match_back],
        [InRange(43, 47), Rejects(40, 43), InRange(15, 19), Rejects(14, 15), Done]
    );

    search_asserts!(
        STRESS,
        '😁',
        "Reverse search for four-byte emoji; check if next_back() still works",
        [next_match_back, next_back, next_match_back, next_back, next_match_back],
        [InRange(43, 47), Rejects(40, 43), InRange(15, 19), Rejects(14, 15), Done]
    );

    search_asserts!(
        STRESS,
        'ꁁ',
        "Reverse search for three-byte Yi character with repeated bytes",
        [next_match_back, next_back, next_match_back, next_back, next_match_back],
        [InRange(37, 40), Rejects(34, 37), InRange(10, 13), Rejects(8, 10), Done]
    );

    search_asserts!(
        STRESS,
        'ꁁ',
        "Reverse search for three-byte Yi character with repeated bytes; check if next_back() still works",
        [next_match_back, next_back, next_match_back, next_back, next_match_back],
        [InRange(37, 40), Rejects(34, 37), InRange(10, 13), Rejects(8, 10), Done]
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
        [next_match, next_match_back, next_match, next_match_back],
        [InRange(0, 1), InRange(10, 11), InRange(5, 6), Done]
    );
    search_asserts!(
        "abcdeabcdeabcde",
        'a',
        "triple double ended search for a",
        [next_match, next_match_back, next_match_back, next_match_back],
        [InRange(0, 1), InRange(10, 11), InRange(5, 6), Done]
    );
    search_asserts!(
        "abcdeabcdeabcde",
        'd',
        "triple double ended search for d",
        [next_match, next_match_back, next_match_back, next_match_back],
        [InRange(3, 4), InRange(13, 14), InRange(8, 9), Done]
    );
    search_asserts!(
        STRESS,
        'Á',
        "Double ended search for two-byte Latin character",
        [next_match, next_match_back, next_match, next_match_back],
        [InRange(0, 2), InRange(32, 34), InRange(8, 10), Done]
    );
    search_asserts!(
        STRESS,
        '각',
        "Reverse double ended search for three-byte Hangul character",
        [next_match_back, next_back, next_match, next, next_match_back, next_match],
        [InRange(34, 37), Rejects(32, 34), InRange(19, 22), Rejects(22, 25), InRange(28, 31), Done]
    );
    search_asserts!(
        STRESS,
        'ก',
        "Double ended search for three-byte Thai character",
        [next_match, next_back, next, next_match_back, next_match],
        [InRange(22, 25), Rejects(47, 48), Rejects(25, 28), InRange(40, 43), Done]
    );
    search_asserts!(
        STRESS,
        '😁',
        "Double ended search for four-byte emoji",
        [next_match_back, next, next_match, next_back, next_match],
        [InRange(43, 47), Rejects(0, 2), InRange(15, 19), Rejects(40, 43), Done]
    );
    search_asserts!(
        STRESS,
        'ꁁ',
        "Double ended search for three-byte Yi character with repeated bytes",
        [next_match, next, next_match_back, next_back, next_match],
        [InRange(10, 13), Rejects(13, 14), InRange(37, 40), Rejects(34, 37), Done]
    );
}
