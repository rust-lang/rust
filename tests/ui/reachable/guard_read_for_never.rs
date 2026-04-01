// Regression test for <https://github.com/rust-lang/rust/pull/149545#discussion_r2585205872>
//
//@ check-pass
#![feature(guard_patterns, never_type)]
#![expect(incomplete_features)]
#![deny(unreachable_code)]

fn main() {
    unsafe {
        let x = std::ptr::null::<!>();

        // This should not constitute a read for never, therefore no code here is unreachable
        let (_ if false): ! = *x;
        ();
    }
}
