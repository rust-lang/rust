// These tests are only run for the "default" test target because some of them
// can take quite a long time. Some of them take long enough that it's not
// practical to run them in debug mode. :-/

// See: https://oss-fuzz.com/testcase-detail/5673225499181056
//
// Ignored by default since it takes too long in debug mode (almost a minute).
#[test]
#[ignore]
fn fuzz1() {
    regex!(r"1}{55}{0}*{1}{55}{55}{5}*{1}{55}+{56}|;**");
}

// See: https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=26505
// See: https://github.com/rust-lang/regex/issues/722
#[test]
fn empty_any_errors_no_panic() {
    assert!(regex_new!(r"\P{any}").is_err());
}

// This tests that a very large regex errors during compilation instead of
// using gratuitous amounts of memory. The specific problem is that the
// compiler wasn't accounting for the memory used by Unicode character classes
// correctly.
//
// See: https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=33579
#[test]
fn big_regex_fails_to_compile() {
    let pat = "[\u{0}\u{e}\u{2}\\w~~>[l\t\u{0}]p?<]{971158}";
    assert!(regex_new!(pat).is_err());
}

// This was caught while on master but before a release went out(!).
//
// See: https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=58173
#[test]
fn todo() {
    let pat = "(?:z|xx)@|xx";
    assert!(regex_new!(pat).is_ok());
}
