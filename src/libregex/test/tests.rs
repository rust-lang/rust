// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength
// ignore-lexer-test FIXME #15679

use regex::{Regex, NoExpand};

#[test]
fn splitn() {
    let re = regex!(r"\d+");
    let text = "cauchy123plato456tyler789binx";
    let subs: Vec<&str> = re.splitn(text, 2).collect();
    assert_eq!(subs, vec!("cauchy", "plato456tyler789binx"));
}

#[test]
fn split() {
    let re = regex!(r"\d+");
    let text = "cauchy123plato456tyler789binx";
    let subs: Vec<&str> = re.split(text).collect();
    assert_eq!(subs, vec!("cauchy", "plato", "tyler", "binx"));
}

#[test]
fn empty_regex_empty_match() {
    let re = regex!("");
    let ms = re.find_iter("").collect::<Vec<(uint, uint)>>();
    assert_eq!(ms, vec![(0, 0)]);
}

#[test]
fn empty_regex_nonempty_match() {
    let re = regex!("");
    let ms = re.find_iter("abc").collect::<Vec<(uint, uint)>>();
    assert_eq!(ms, vec![(0, 0), (1, 1), (2, 2), (3, 3)]);
}

macro_rules! replace(
    ($name:ident, $which:ident, $re:expr,
     $search:expr, $replace:expr, $result:expr) => (
        #[test]
        fn $name() {
            let re = regex!($re);
            assert_eq!(re.$which($search, $replace), String::from_str($result));
        }
    );
)

replace!(rep_first, replace, r"\d", "age: 26", "Z", "age: Z6")
replace!(rep_plus, replace, r"\d+", "age: 26", "Z", "age: Z")
replace!(rep_all, replace_all, r"\d", "age: 26", "Z", "age: ZZ")
replace!(rep_groups, replace, r"(\S+)\s+(\S+)", "w1 w2", "$2 $1", "w2 w1")
replace!(rep_double_dollar, replace,
         r"(\S+)\s+(\S+)", "w1 w2", "$2 $$1", "w2 $1")
replace!(rep_no_expand, replace,
         r"(\S+)\s+(\S+)", "w1 w2", NoExpand("$2 $1"), "$2 $1")
replace!(rep_named, replace_all,
         r"(?P<first>\S+)\s+(?P<last>\S+)(?P<space>\s*)",
         "w1 w2 w3 w4", "$last $first$space", "w2 w1 w4 w3")
replace!(rep_trim, replace_all, "^[ \t]+|[ \t]+$", " \t  trim me\t   \t",
         "", "trim me")

macro_rules! noparse(
    ($name:ident, $re:expr) => (
        #[test]
        fn $name() {
            let re = $re;
            match Regex::new(re) {
                Err(_) => {},
                Ok(_) => fail!("Regex '{}' should cause a parse error.", re),
            }
        }
    );
)

noparse!(fail_double_repeat, "a**")
noparse!(fail_no_repeat_arg, "*")
noparse!(fail_no_repeat_arg_begin, "^*")
noparse!(fail_incomplete_escape, "\\")
noparse!(fail_class_incomplete, "[A-")
noparse!(fail_class_not_closed, "[A")
noparse!(fail_class_no_begin, r"[\A]")
noparse!(fail_class_no_end, r"[\z]")
noparse!(fail_class_no_boundary, r"[\b]")
noparse!(fail_open_paren, "(")
noparse!(fail_close_paren, ")")
noparse!(fail_invalid_range, "[a-Z]")
noparse!(fail_empty_capture_name, "(?P<>a)")
noparse!(fail_empty_capture_exp, "(?P<name>)")
noparse!(fail_bad_capture_name, "(?P<na-me>)")
noparse!(fail_bad_flag, "(?a)a")
noparse!(fail_empty_alt_before, "|a")
noparse!(fail_empty_alt_after, "a|")
noparse!(fail_counted_big_exact, "a{1001}")
noparse!(fail_counted_big_min, "a{1001,}")
noparse!(fail_counted_no_close, "a{1001")
noparse!(fail_unfinished_cap, "(?")
noparse!(fail_unfinished_escape, "\\")
noparse!(fail_octal_digit, r"\8")
noparse!(fail_hex_digit, r"\xG0")
noparse!(fail_hex_short, r"\xF")
noparse!(fail_hex_long_digits, r"\x{fffg}")
noparse!(fail_flag_bad, "(?a)")
noparse!(fail_flag_empty, "(?)")
noparse!(fail_double_neg, "(?-i-i)")
noparse!(fail_neg_empty, "(?i-)")
noparse!(fail_empty_group, "()")
noparse!(fail_dupe_named, "(?P<a>.)(?P<a>.)")

macro_rules! mat(
    ($name:ident, $re:expr, $text:expr, $($loc:tt)+) => (
        #[test]
        fn $name() {
            let text = $text;
            let expected: Vec<Option<(uint, uint)>> = vec!($($loc)+);
            let r = regex!($re);
            let got = match r.captures(text) {
                Some(c) => c.iter_pos().collect::<Vec<Option<(uint, uint)>>>(),
                None => vec!(None),
            };
            // The test set sometimes leave out capture groups, so truncate
            // actual capture groups to match test set.
            let (sexpect, mut sgot) = (expected.as_slice(), got.as_slice());
            if sgot.len() > sexpect.len() {
                sgot = sgot[0..sexpect.len()]
            }
            if sexpect != sgot {
                fail!("For RE '{}' against '{}', expected '{}' but got '{}'",
                      $re, text, sexpect, sgot);
            }
        }
    );
)

// Some crazy expressions from regular-expressions.info.
mat!(match_ranges,
     r"\b(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])\b",
     "num: 255", Some((5, 8)))
mat!(match_ranges_not,
     r"\b(?:[0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-4][0-9]|25[0-5])\b",
     "num: 256", None)
mat!(match_float1, r"[-+]?[0-9]*\.?[0-9]+", "0.1", Some((0, 3)))
mat!(match_float2, r"[-+]?[0-9]*\.?[0-9]+", "0.1.2", Some((0, 3)))
mat!(match_float3, r"[-+]?[0-9]*\.?[0-9]+", "a1.2", Some((1, 4)))
mat!(match_float4, r"^[-+]?[0-9]*\.?[0-9]+$", "1.a", None)
mat!(match_email, r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}\b",
     "mine is jam.slam@gmail.com ", Some((8, 26)))
mat!(match_email_not, r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}\b",
     "mine is jam.slam@gmail ", None)
mat!(match_email_big, r"[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?",
     "mine is jam.slam@gmail.com ", Some((8, 26)))
mat!(match_date1,
     r"^(19|20)\d\d[- /.](0[1-9]|1[012])[- /.](0[1-9]|[12][0-9]|3[01])$",
     "1900-01-01", Some((0, 10)))
mat!(match_date2,
     r"^(19|20)\d\d[- /.](0[1-9]|1[012])[- /.](0[1-9]|[12][0-9]|3[01])$",
     "1900-00-01", None)
mat!(match_date3,
     r"^(19|20)\d\d[- /.](0[1-9]|1[012])[- /.](0[1-9]|[12][0-9]|3[01])$",
     "1900-13-01", None)

// Exercise the flags.
mat!(match_flag_case, "(?i)abc", "ABC", Some((0, 3)))
mat!(match_flag_weird_case, "(?i)a(?-i)bc", "Abc", Some((0, 3)))
mat!(match_flag_weird_case_not, "(?i)a(?-i)bc", "ABC", None)
mat!(match_flag_case_dotnl, "(?is)a.", "A\n", Some((0, 2)))
mat!(match_flag_case_dotnl_toggle, "(?is)a.(?-is)a.", "A\nab", Some((0, 4)))
mat!(match_flag_case_dotnl_toggle_not, "(?is)a.(?-is)a.", "A\na\n", None)
mat!(match_flag_case_dotnl_toggle_ok, "(?is)a.(?-is:a.)?", "A\na\n", Some((0, 2)))
mat!(match_flag_multi, "(?m)(?:^\\d+$\n?)+", "123\n456\n789", Some((0, 11)))
mat!(match_flag_ungreedy, "(?U)a+", "aa", Some((0, 1)))
mat!(match_flag_ungreedy_greedy, "(?U)a+?", "aa", Some((0, 2)))
mat!(match_flag_ungreedy_noop, "(?U)(?-U)a+", "aa", Some((0, 2)))

// Some Unicode tests.
mat!(uni_literal, r"Ⅰ", "Ⅰ", Some((0, 3)))
mat!(uni_one, r"\pN", "Ⅰ", Some((0, 3)))
mat!(uni_mixed, r"\pN+", "Ⅰ1Ⅱ2", Some((0, 8)))
mat!(uni_not, r"\PN+", "abⅠ", Some((0, 2)))
mat!(uni_not_class, r"[\PN]+", "abⅠ", Some((0, 2)))
mat!(uni_not_class_neg, r"[^\PN]+", "abⅠ", Some((2, 5)))
mat!(uni_case, r"(?i)Δ", "δ", Some((0, 2)))
mat!(uni_case_not, r"Δ", "δ", None)
mat!(uni_case_upper, r"\p{Lu}+", "ΛΘΓΔα", Some((0, 8)))
mat!(uni_case_upper_nocase_flag, r"(?i)\p{Lu}+", "ΛΘΓΔα", Some((0, 10)))
mat!(uni_case_upper_nocase, r"\p{L}+", "ΛΘΓΔα", Some((0, 10)))
mat!(uni_case_lower, r"\p{Ll}+", "ΛΘΓΔα", Some((8, 10)))

// Test the Unicode friendliness of Perl character classes.
mat!(uni_perl_w, r"\w+", "dδd", Some((0, 4)))
mat!(uni_perl_w_not, r"\w+", "⥡", None)
mat!(uni_perl_w_neg, r"\W+", "⥡", Some((0, 3)))
mat!(uni_perl_d, r"\d+", "1२३9", Some((0, 8)))
mat!(uni_perl_d_not, r"\d+", "Ⅱ", None)
mat!(uni_perl_d_neg, r"\D+", "Ⅱ", Some((0, 3)))
mat!(uni_perl_s, r"\s+", " ", Some((0, 3)))
mat!(uni_perl_s_not, r"\s+", "☃", None)
mat!(uni_perl_s_neg, r"\S+", "☃", Some((0, 3)))

// And do the same for word boundaries.
mat!(uni_boundary_none, r"\d\b", "6δ", None)
mat!(uni_boundary_ogham, r"\d\b", "6 ", Some((0, 1)))

// A whole mess of tests from Glenn Fowler's regex test suite.
// Generated by the 'src/etc/regex-match-tests' program.
mod matches;
