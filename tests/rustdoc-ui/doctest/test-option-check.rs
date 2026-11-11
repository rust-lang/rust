//@ compile-flags: --test
//@ check-test-line-numbers-match
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

/// This looks like another awesome test!
///
/// ```
/// println!("foo?");
/// ```
pub fn foooo() {}
