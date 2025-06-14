//! Regression test for issue #1818
//! last-use analysis in closures should allow moves instead of requiring copies.
//!
//! The original issue was that the compiler incorrectly flagged certain return values
//! in anonymous functions/closures as requiring copies of non-copyable values, when
//! they should have been treated as moves (since they were the last use of the value).
//!
//! See: https://github.com/rust-lang/rust/issues/1818

//@ run-pass

fn apply<T, F>(s: String, mut f: F) -> T
where
    F: FnMut(String) -> T
{
    fn g<T, F>(s: String, mut f: F) -> T
    where
        F: FnMut(String) -> T
    {
        f(s)
    }

    g(s, |v| {
        let r = f(v);
        r // This should be a move, not requiring copy
    })
}

pub fn main() {
    // Actually test the functionality
    let result = apply(String::from("test"), |s| s.len());
    assert_eq!(result, 4);
}
