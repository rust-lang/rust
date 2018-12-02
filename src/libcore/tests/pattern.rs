use core::needle::*;

// This macro makes it easier to write
// tests that do a series of iterations
macro_rules! search_asserts {
    ($haystack:expr, $needle:expr, $testname:expr, [$($op:ident = $expected:expr,)*]) => {
        let mut searcher = ext::match_ranges($haystack, $needle).map(|(r, _)| r);
        let actual = [$(searcher.$op()),*];
        assert_eq!(&actual[..], &[$($expected),*][..], $testname);
    };
}

#[test]
fn test_simple_iteration() {
    search_asserts! ("abcdeabcd", 'a', "forward iteration for ASCII string", [
        next = Some(0..1),
        next = Some(5..6),
        next = None,
    ]);

    search_asserts! ("abcdeabcd", 'a', "reverse iteration for ASCII string", [
        next_back = Some(5..6),
        next_back = Some(0..1),
        next_back = None,
    ]);

    search_asserts! ("我爱我的猫", '我', "forward iteration for Chinese string", [
        next = Some(0..3),
        next = Some(6..9),
        next = None,
    ]);

    search_asserts! ("我的猫说meow", 'm', "forward iteration for mixed string", [
        next = Some(12..13),
        next = None,
    ]);

    search_asserts! ("我的猫说meow", '猫', "reverse iteration for mixed string", [
        next_back = Some(6..9),
        next_back = None,
    ]);
}

#[test]
fn test_simple_search() {
    search_asserts!("abcdeabcdeabcde", 'a', "next_match for ASCII string", [
        next = Some(0..1),
        next = Some(5..6),
        next = Some(10..11),
        next = None,
    ]);

    search_asserts!("abcdeabcdeabcde", 'a', "next_match_back for ASCII string", [
        next_back = Some(10..11),
        next_back = Some(5..6),
        next_back = Some(0..1),
        next_back = None,
    ]);
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
fn test_forward_search_shared_bytes() {
    search_asserts!(STRESS, 'Á', "Forward search for two-byte Latin character", [
        next = Some(0..2),
        next = Some(8..10),
        next = Some(32..34),
        next = None,
    ]);

    search_asserts!(STRESS, '각', "Forward search for three-byte Hangul character", [
        next = Some(19..22),
        next = Some(28..31),
        next = Some(34..37),
        next = None,
    ]);

    search_asserts!(STRESS, 'ก', "Forward search for three-byte Thai character", [
        next = Some(22..25),
        next = Some(40..43),
        next = None,
    ]);

    search_asserts!(STRESS, '😁', "Forward search for four-byte emoji", [
        next = Some(15..19),
        next = Some(43..47),
        next = None,
    ]);

    search_asserts!(STRESS, 'ꁁ', "Forward search for three-byte Yi character with repeated bytes", [
        next = Some(10..13),
        next = Some(37..40),
        next = None,
    ]);
}

#[test]
fn test_reverse_search_shared_bytes() {
    search_asserts!(STRESS, 'Á', "Reverse search for two-byte Latin character", [
        next_back = Some(32..34),
        next_back = Some(8..10),
        next_back = Some(0..2),
        next_back = None,
    ]);

    search_asserts!(STRESS, '각', "Reverse search for three-byte Hangul character", [
        next_back = Some(34..37),
        next_back = Some(28..31),
        next_back = Some(19..22),
        next_back = None,
    ]);

    search_asserts!(STRESS, 'ก', "Reverse search for three-byte Thai character", [
        next_back = Some(40..43),
        next_back = Some(22..25),
        next_back = None,
    ]);

    search_asserts!(STRESS, '😁', "Reverse search for four-byte emoji", [
        next_back = Some(43..47),
        next_back = Some(15..19),
        next_back = None,
    ]);

    search_asserts!(STRESS, 'ꁁ', "Reverse search for three-byte Yi character with repeated bytes", [
        next_back = Some(37..40),
        next_back = Some(10..13),
        next_back = None,
    ]);
}

#[test]
fn double_ended_regression_test() {
    search_asserts!("abcdeabcdeabcde", 'a', "alternating double ended search", [
        next = Some(0..1),
        next_back = Some(10..11),
        next = Some(5..6),
        next_back = None,
    ]);

    search_asserts!("abcdeabcdeabcde", 'a', "triple double ended search for a", [
        next = Some(0..1),
        next_back = Some(10..11),
        next_back = Some(5..6),
        next_back = None,
    ]);

    search_asserts!("abcdeabcdeabcde", 'd', "triple double ended search for d", [
        next = Some(3..4),
        next_back = Some(13..14),
        next_back = Some(8..9),
        next_back = None,
    ]);

    search_asserts!(STRESS, 'Á', "Double ended search for two-byte Latin character", [
        next = Some(0..2),
        next_back = Some(32..34),
        next = Some(8..10),
        next_back = None,
    ]);

    search_asserts!(STRESS, '각', "Reverse double ended search for three-byte Hangul character", [
        next_back = Some(34..37),
        next = Some(19..22),
        next_back = Some(28..31),
        next = None,
    ]);

    search_asserts!(STRESS, 'ก', "Double ended search for three-byte Thai character", [
        next = Some(22..25),
        next_back = Some(40..43),
        next = None,
    ]);

    search_asserts!(STRESS, '😁', "Double ended search for four-byte emoji", [
        next_back = Some(43..47),
        next = Some(15..19),
        next = None,
    ]);

    search_asserts!(STRESS, 'ꁁ', "Double ended search for 3-byte Yi char with repeated bytes", [
        next = Some(10..13),
        next_back = Some(37..40),
        next = None,
    ]);
}

#[test]
fn test_stress_indices() {
    // this isn't really a test,
    // more of documentation on the indices of each character in the stresstest string

    search_asserts!(STRESS, |_: char| true, "Indices of characters in stress test", [
        next = Some(0..2), // Á
        next = Some(2..3), // a
        next = Some(3..7), // 🁀
        next = Some(7..8), // b
        next = Some(8..10), // Á
        next = Some(10..13), // ꁁ
        next = Some(13..14), // f
        next = Some(14..15), // g
        next = Some(15..19), // 😀
        next = Some(19..22), // 각
        next = Some(22..25), // ก
        next = Some(25..28), // ᘀ
        next = Some(28..31), // 각
        next = Some(31..32), // a
        next = Some(32..34), // Á
        next = Some(34..37), // 각
        next = Some(37..40), // ꁁ
        next = Some(40..43), // ก
        next = Some(43..47), // 😀
        next = Some(47..48), // a
        next = None,
    ]);

    search_asserts!(STRESS, |_: char| true, "Indices of characters in stress test, reversed", [
        next_back = Some(47..48), // a
        next_back = Some(43..47), // 😀
        next_back = Some(40..43), // ก
        next_back = Some(37..40), // ꁁ
        next_back = Some(34..37), // 각
        next_back = Some(32..34), // Á
        next_back = Some(31..32), // a
        next_back = Some(28..31), // 각
        next_back = Some(25..28), // ᘀ
        next_back = Some(22..25), // ก
        next_back = Some(19..22), // 각
        next_back = Some(15..19), // 😀
        next_back = Some(14..15), // g
        next_back = Some(13..14), // f
        next_back = Some(10..13), // ꁁ
        next_back = Some(8..10), // Á
        next_back = Some(7..8), // b
        next_back = Some(3..7), // 🁀
        next_back = Some(2..3), // a
        next_back = Some(0..2), // Á
        next_back = None,
    ]);
}

#[test]
fn test_fn_double_ended() {
    search_asserts!(
        STRESS,
        |c: char| c >= '\u{10000}',
        "Search for all non-BMP characters, double ended",
        [
            next = Some(3..7),
            next_back = Some(43..47),
            next = Some(15..19),
            next_back = None,
            next = None,
        ]
    );
}

#[test]
fn test_str() {
    search_asserts!("abbcbbd", "bb", "str_searcher_ascii_haystack::fwd", [
        next = Some(1..3),
        next = Some(4..6),
        next = None,
    ]);

    search_asserts!("abbcbbbbd", "bb", "str_searcher_ascii_haystack_seq::fwd", [
        next = Some(1..3),
        next = Some(4..6),
        next = Some(6..8),
        next = None,
    ]);

    search_asserts!("abbcbbd", "", "str_searcher_empty_needle_ascii_haystack::fwd", [
        next = Some(0..0),
        next = Some(1..1),
        next = Some(2..2),
        next = Some(3..3),
        next = Some(4..4),
        next = Some(5..5),
        next = Some(6..6),
        next = Some(7..7),
        next = None,
    ]);

    search_asserts!("├──", " ", "str_searcher_multibyte_haystack::fwd", [
        next = None,
    ]);

    search_asserts!("├──", "", "str_searcher_empty_needle_multibyte_haystack::fwd", [
        next = Some(0..0),
        next = Some(3..3),
        next = Some(6..6),
        next = Some(9..9),
        next = None,
    ]);

    search_asserts!("", "", "str_searcher_empty_needle_multibyte_haystack::fwd", [
        next = Some(0..0),
        next = None,
    ]);
}
