#![cfg_attr(feature = "pattern", feature(pattern))]

use regex;

// Due to macro scoping rules, this definition only applies for the modules
// defined below. Effectively, it allows us to use the same tests for both
// native and dynamic regexes.
//
// This is also used to test the various matching engines. This one exercises
// the normal code path which automatically chooses the engine based on the
// regex and the input. Other dynamic tests explicitly set the engine to use.
macro_rules! regex_new {
    ($re:expr) => {{
        use regex::Regex;
        Regex::new($re)
    }};
}

macro_rules! regex {
    ($re:expr) => {
        regex_new!($re).unwrap()
    };
}

macro_rules! regex_set_new {
    ($re:expr) => {{
        use regex::RegexSet;
        RegexSet::new($re)
    }};
}

macro_rules! regex_set {
    ($res:expr) => {
        regex_set_new!($res).unwrap()
    };
}

// Must come before other module definitions.
include!("macros_str.rs");
include!("macros.rs");

mod api;
mod api_str;
mod crazy;
mod flags;
mod fowler;
mod misc;
mod multiline;
mod noparse;
mod regression;
mod regression_fuzz;
mod replace;
mod searcher;
mod set;
mod shortest_match;
mod suffix_reverse;
#[cfg(feature = "unicode")]
mod unicode;
#[cfg(feature = "unicode-perl")]
mod word_boundary;
#[cfg(feature = "unicode-perl")]
mod word_boundary_unicode;

#[test]
fn disallow_non_utf8() {
    assert!(regex::Regex::new(r"(?-u)\xFF").is_err());
    assert!(regex::Regex::new(r"(?-u).").is_err());
    assert!(regex::Regex::new(r"(?-u)[\xFF]").is_err());
    assert!(regex::Regex::new(r"(?-u)☃").is_err());
}

#[test]
fn disallow_octal() {
    assert!(regex::Regex::new(r"\0").is_err());
}

#[test]
fn allow_octal() {
    assert!(regex::RegexBuilder::new(r"\0").octal(true).build().is_ok());
}

#[test]
fn oibits() {
    use regex::bytes;
    use regex::{Regex, RegexBuilder, RegexSet, RegexSetBuilder};
    use std::panic::{RefUnwindSafe, UnwindSafe};

    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    fn assert_unwind_safe<T: UnwindSafe>() {}
    fn assert_ref_unwind_safe<T: RefUnwindSafe>() {}

    assert_send::<Regex>();
    assert_sync::<Regex>();
    assert_unwind_safe::<Regex>();
    assert_ref_unwind_safe::<Regex>();
    assert_send::<RegexBuilder>();
    assert_sync::<RegexBuilder>();
    assert_unwind_safe::<RegexBuilder>();
    assert_ref_unwind_safe::<RegexBuilder>();

    assert_send::<bytes::Regex>();
    assert_sync::<bytes::Regex>();
    assert_unwind_safe::<bytes::Regex>();
    assert_ref_unwind_safe::<bytes::Regex>();
    assert_send::<bytes::RegexBuilder>();
    assert_sync::<bytes::RegexBuilder>();
    assert_unwind_safe::<bytes::RegexBuilder>();
    assert_ref_unwind_safe::<bytes::RegexBuilder>();

    assert_send::<RegexSet>();
    assert_sync::<RegexSet>();
    assert_unwind_safe::<RegexSet>();
    assert_ref_unwind_safe::<RegexSet>();
    assert_send::<RegexSetBuilder>();
    assert_sync::<RegexSetBuilder>();
    assert_unwind_safe::<RegexSetBuilder>();
    assert_ref_unwind_safe::<RegexSetBuilder>();

    assert_send::<bytes::RegexSet>();
    assert_sync::<bytes::RegexSet>();
    assert_unwind_safe::<bytes::RegexSet>();
    assert_ref_unwind_safe::<bytes::RegexSet>();
    assert_send::<bytes::RegexSetBuilder>();
    assert_sync::<bytes::RegexSetBuilder>();
    assert_unwind_safe::<bytes::RegexSetBuilder>();
    assert_ref_unwind_safe::<bytes::RegexSetBuilder>();
}

// See: https://github.com/rust-lang/regex/issues/568
#[test]
fn oibits_regression() {
    use regex::Regex;
    use std::panic;

    let _ = panic::catch_unwind(|| Regex::new("a").unwrap());
}

// See: https://github.com/rust-lang/regex/issues/750
#[test]
#[cfg(target_pointer_width = "64")]
fn regex_is_reasonably_small() {
    use std::mem::size_of;

    use regex::bytes;
    use regex::{Regex, RegexSet};

    assert_eq!(16, size_of::<Regex>());
    assert_eq!(16, size_of::<RegexSet>());
    assert_eq!(16, size_of::<bytes::Regex>());
    assert_eq!(16, size_of::<bytes::RegexSet>());
}

// See: https://github.com/rust-lang/regex/security/advisories/GHSA-m5pq-gvj9-9vr8
// See: CVE-2022-24713
//
// We test that our regex compiler will correctly return a "too big" error when
// we try to use a very large repetition on an *empty* sub-expression.
//
// At the time this test was written, the regex compiler does not represent
// empty sub-expressions with any bytecode instructions. In effect, it's an
// "optimization" to leave them out, since they would otherwise correspond
// to an unconditional JUMP in the regex bytecode (i.e., an unconditional
// epsilon transition in the NFA graph). Therefore, an empty sub-expression
// represents an interesting case for the compiler's size limits. Since it
// doesn't actually contribute any additional memory to the compiled regex
// instructions, the size limit machinery never detects it. Instead, it just
// dumbly tries to compile the empty sub-expression N times, where N is the
// repetition size.
//
// When N is very large, this will cause the compiler to essentially spin and
// do nothing for a decently large amount of time. It causes the regex to take
// quite a bit of time to compile, despite the concrete syntax of the regex
// being quite small.
//
// The degree to which this is actually a problem is somewhat of a judgment
// call. Some regexes simply take a long time to compile. But in general, you
// should be able to reasonably control this by setting lower or higher size
// limits on the compiled object size. But this mitigation doesn't work at all
// for this case.
//
// This particular test is somewhat narrow. It merely checks that regex
// compilation will, at some point, return a "too big" error. Before the
// fix landed, this test would eventually fail because the regex would be
// successfully compiled (after enough time elapsed). So while this test
// doesn't check that we exit in a reasonable amount of time, it does at least
// check that we are properly returning an error at some point.
#[test]
fn big_empty_regex_fails() {
    use regex::Regex;

    let result = Regex::new("(?:){4294967295}");
    assert!(result.is_err());
}

// Below is a "billion laughs" variant of the previous test case.
#[test]
fn big_empty_reps_chain_regex_fails() {
    use regex::Regex;

    let result = Regex::new("(?:){64}{64}{64}{64}{64}{64}");
    assert!(result.is_err());
}

// Below is another situation where a zero-length sub-expression can be
// introduced.
#[test]
fn big_zero_reps_regex_fails() {
    use regex::Regex;

    let result = Regex::new(r"x{0}{4294967295}");
    assert!(result.is_err());
}

// Testing another case for completeness.
#[test]
fn empty_alt_regex_fails() {
    use regex::Regex;

    let result = Regex::new(r"(?:|){4294967295}");
    assert!(result.is_err());
}

// Regression test for: https://github.com/rust-lang/regex/issues/969
#[test]
fn regression_i969() {
    use regex::Regex;

    let re = Regex::new(r"c.*d\z").unwrap();
    assert_eq!(Some(6), re.shortest_match_at("ababcd", 4));
    assert_eq!(Some(6), re.find_at("ababcd", 4).map(|m| m.end()));
}
