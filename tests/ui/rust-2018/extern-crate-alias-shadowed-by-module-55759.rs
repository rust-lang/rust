// Regression test for <https://github.com/rust-lang/rust/issues/55759>.
// `rust_2018_idioms` used to suggest rewriting this `extern crate` as `use time as
// std_time;`. Because a local module shares the crate's name, the rewritten import is
// ambiguous (E0659), so applying the suggestion produced code that did not compile.
//@ edition: 2018
//@ aux-build: time.rs
//@ check-pass

#![warn(rust_2018_idioms)]

extern crate time as std_time;

pub mod time {
    pub fn f() {
        let _ = crate::std_time::now();
    }
}

fn main() {}
