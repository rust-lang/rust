// See: https://github.com/rust-lang/regex/issues/48
#[test]
fn invalid_regexes_no_crash() {
    assert!(regex_new!("(*)").is_err());
    assert!(regex_new!("(?:?)").is_err());
    assert!(regex_new!("(?)").is_err());
    assert!(regex_new!("*").is_err());
}

// See: https://github.com/rust-lang/regex/issues/98
#[test]
fn regression_many_repeat_stack_overflow() {
    let re = regex!("^.{1,2500}");
    assert_eq!(vec![(0, 1)], findall!(re, "a"));
}

// See: https://github.com/rust-lang/regex/issues/555
#[test]
fn regression_invalid_repetition_expr() {
    assert!(regex_new!("(?m){1,1}").is_err());
}

// See: https://github.com/rust-lang/regex/issues/527
#[test]
fn regression_invalid_flags_expression() {
    assert!(regex_new!("(((?x)))").is_ok());
}

// See: https://github.com/rust-lang/regex/issues/75
mat!(regression_unsorted_binary_search_1, r"(?i-u)[a_]+", "A_", Some((0, 2)));
mat!(regression_unsorted_binary_search_2, r"(?i-u)[A_]+", "a_", Some((0, 2)));

// See: https://github.com/rust-lang/regex/issues/99
#[cfg(feature = "unicode-case")]
mat!(regression_negated_char_class_1, r"(?i)[^x]", "x", None);
#[cfg(feature = "unicode-case")]
mat!(regression_negated_char_class_2, r"(?i)[^x]", "X", None);

// See: https://github.com/rust-lang/regex/issues/101
mat!(regression_ascii_word_underscore, r"[[:word:]]", "_", Some((0, 1)));

// See: https://github.com/rust-lang/regex/issues/129
#[test]
fn regression_captures_rep() {
    let re = regex!(r"([a-f]){2}(?P<foo>[x-z])");
    let caps = re.captures(text!("abx")).unwrap();
    assert_eq!(match_text!(caps.name("foo").unwrap()), text!("x"));
}

// See: https://github.com/rust-lang/regex/issues/153
mat!(regression_alt_in_alt1, r"ab?|$", "az", Some((0, 1)));
mat!(regression_alt_in_alt2, r"^(.*?)(\n|\r\n?|$)", "ab\rcd", Some((0, 3)));

// See: https://github.com/rust-lang/regex/issues/169
mat!(regression_leftmost_first_prefix, r"z*azb", "azb", Some((0, 3)));

// See: https://github.com/rust-lang/regex/issues/76
#[cfg(all(feature = "unicode-case", feature = "unicode-gencat"))]
mat!(uni_case_lower_nocase_flag, r"(?i)\p{Ll}+", "ΛΘΓΔα", Some((0, 10)));

// See: https://github.com/rust-lang/regex/issues/191
mat!(many_alternates, r"1|2|3|4|5|6|7|8|9|10|int", "int", Some((0, 3)));

// burntsushi was bad and didn't create an issue for this bug.
mat!(anchored_prefix1, r"^a[[:^space:]]", "a ", None);
mat!(anchored_prefix2, r"^a[[:^space:]]", "foo boo a ", None);
mat!(anchored_prefix3, r"^-[a-z]", "r-f", None);

// See: https://github.com/rust-lang/regex/issues/204
#[cfg(feature = "unicode-perl")]
split!(
    split_on_word_boundary,
    r"\b",
    r"Should this (work?)",
    &[
        t!(""),
        t!("Should"),
        t!(" "),
        t!("this"),
        t!(" ("),
        t!("work"),
        t!("?)")
    ]
);
#[cfg(feature = "unicode-perl")]
matiter!(
    word_boundary_dfa,
    r"\b",
    "a b c",
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5)
);

// See: https://github.com/rust-lang/regex/issues/268
matiter!(partial_anchor, r"^a|b", "ba", (0, 1));

// See: https://github.com/rust-lang/regex/issues/280
ismatch!(partial_anchor_alternate_begin, r"^a|z", "yyyyya", false);
ismatch!(partial_anchor_alternate_end, r"a$|z", "ayyyyy", false);

// See: https://github.com/rust-lang/regex/issues/289
mat!(lits_unambiguous1, r"(ABC|CDA|BC)X", "CDAX", Some((0, 4)));

// See: https://github.com/rust-lang/regex/issues/291
mat!(
    lits_unambiguous2,
    r"((IMG|CAM|MG|MB2)_|(DSCN|CIMG))(?P<n>[0-9]+)$",
    "CIMG2341",
    Some((0, 8)),
    Some((0, 4)),
    None,
    Some((0, 4)),
    Some((4, 8))
);

// See: https://github.com/rust-lang/regex/issues/271
mat!(endl_or_wb, r"(?m:$)|(?-u:\b)", "\u{6084e}", Some((4, 4)));
mat!(zero_or_end, r"(?i-u:\x00)|$", "\u{e682f}", Some((4, 4)));
mat!(y_or_endl, r"(?i-u:y)|(?m:$)", "\u{b4331}", Some((4, 4)));
#[cfg(feature = "unicode-perl")]
mat!(wb_start_x, r"(?u:\b)^(?-u:X)", "X", Some((0, 1)));

// See: https://github.com/rust-lang/regex/issues/321
ismatch!(strange_anchor_non_complete_prefix, r"a^{2}", "", false);
ismatch!(strange_anchor_non_complete_suffix, r"${2}a", "", false);

// See: https://github.com/BurntSushi/ripgrep/issues/1203
ismatch!(reverse_suffix1, r"[0-4][0-4][0-4]000", "153.230000", true);
ismatch!(reverse_suffix2, r"[0-9][0-9][0-9]000", "153.230000\n", true);
matiter!(reverse_suffix3, r"[0-9][0-9][0-9]000", "153.230000\n", (4, 10));

// See: https://github.com/rust-lang/regex/issues/334
// See: https://github.com/rust-lang/regex/issues/557
mat!(
    captures_after_dfa_premature_end1,
    r"a(b*(X|$))?",
    "abcbX",
    Some((0, 1)),
    None,
    None
);
mat!(
    captures_after_dfa_premature_end2,
    r"a(bc*(X|$))?",
    "abcbX",
    Some((0, 1)),
    None,
    None
);
mat!(captures_after_dfa_premature_end3, r"(aa$)?", "aaz", Some((0, 0)));

// See: https://github.com/rust-lang/regex/issues/437
ismatch!(
    literal_panic,
    r"typename type\-parameter\-[0-9]+\-[0-9]+::.+",
    "test",
    false
);

// See: https://github.com/rust-lang/regex/issues/533
ismatch!(
    blank_matches_nothing_between_space_and_tab,
    r"[[:blank:]]",
    "\u{a}\u{b}\u{c}\u{d}\u{e}\u{f}\
     \u{10}\u{11}\u{12}\u{13}\u{14}\u{15}\u{16}\u{17}\
     \u{18}\u{19}\u{1a}\u{1b}\u{1c}\u{1d}\u{1e}\u{1f}",
    false
);

ismatch!(
    inverted_blank_matches_everything_between_space_and_tab,
    r"^[[:^blank:]]+$",
    "\u{a}\u{b}\u{c}\u{d}\u{e}\u{f}\
     \u{10}\u{11}\u{12}\u{13}\u{14}\u{15}\u{16}\u{17}\
     \u{18}\u{19}\u{1a}\u{1b}\u{1c}\u{1d}\u{1e}\u{1f}",
    true
);

// Tests that our Aho-Corasick optimization works correctly. It only
// kicks in when we have >32 literals. By "works correctly," we mean that
// leftmost-first match semantics are properly respected. That is, samwise
// should match, not sam.
mat!(
    ahocorasick1,
    "samwise|sam|a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|w|x|y|z|\
     A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q|R|S|T|U|V|W|X|Y|Z",
    "samwise",
    Some((0, 7))
);

// See: https://github.com/BurntSushi/ripgrep/issues/1247
#[test]
#[cfg(feature = "unicode-perl")]
fn regression_nfa_stops1() {
    let re = ::regex::bytes::Regex::new(r"\bs(?:[ab])").unwrap();
    assert_eq!(0, re.find_iter(b"s\xE4").count());
}

// See: https://github.com/rust-lang/regex/issues/640
#[cfg(feature = "unicode-case")]
matiter!(
    flags_are_unset,
    r"((?i)foo)|Bar",
    "foo Foo bar Bar",
    (0, 3),
    (4, 7),
    (12, 15)
);

// See: https://github.com/rust-lang/regex/issues/659
//
// Note that 'Ј' is not 'j', but cyrillic Je
// https://en.wikipedia.org/wiki/Je_(Cyrillic)
ismatch!(empty_group_match, r"()Ј01", "zЈ01", true);
matiter!(empty_group_find, r"()Ј01", "zЈ01", (1, 5));

// See: https://github.com/rust-lang/regex/issues/862
mat!(non_greedy_question_literal, r"ab??", "ab", Some((0, 1)));

// See: https://github.com/rust-lang/regex/issues/981
#[cfg(feature = "unicode")]
#[test]
fn regression_bad_word_boundary() {
    let re = regex_new!(r#"(?i:(?:\b|_)win(?:32|64|dows)?(?:\b|_))"#).unwrap();
    let hay = "ubi-Darwin-x86_64.tar.gz";
    assert!(!re.is_match(text!(hay)));
    let hay = "ubi-Windows-x86_64.zip";
    assert!(re.is_match(text!(hay)));
}

// See: https://github.com/rust-lang/regex/issues/982
#[cfg(feature = "unicode-perl")]
#[test]
fn regression_unicode_perl_not_enabled() {
    let pat = r"(\d+\s?(years|year|y))?\s?(\d+\s?(months|month|m))?\s?(\d+\s?(weeks|week|w))?\s?(\d+\s?(days|day|d))?\s?(\d+\s?(hours|hour|h))?";
    let re = regex_new!(pat);
    assert!(re.is_ok());
}

// See: https://github.com/rust-lang/regex/issues/995
#[test]
fn regression_big_regex_overflow() {
    let pat = r" {2147483516}{2147483416}{5}";
    let re = regex_new!(pat);
    assert!(re.is_err());
}

#[test]
fn regression_complete_literals_suffix_incorrect() {
    let needles = vec![
        "aA", "bA", "cA", "dA", "eA", "fA", "gA", "hA", "iA", "jA", "kA",
        "lA", "mA", "nA", "oA", "pA", "qA", "rA", "sA", "tA", "uA", "vA",
        "wA", "xA", "yA", "zA",
    ];
    let pattern = needles.join("|");
    let re = regex!(&pattern);
    let hay = "FUBAR";
    assert_eq!(0, re.find_iter(text!(hay)).count());
}
