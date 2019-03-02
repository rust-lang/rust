//! Support code for rustc's built in unit-test and micro-benchmarking
//! framework.
//!
//! Almost all user code will only be interested in `Bencher` and
//! `black_box`. All other interactions (such as writing tests and
//! benchmarks themselves) should be done via the `#[test]` and
//! `#[bench]` attributes.
//!
//! See the [Testing Chapter](../book/ch11-00-testing.html) of the book for more details.

#![crate_name = "test"]
#![unstable(feature = "test", issue = "27812")]
#![doc(html_root_url = "https://doc.rust-lang.org/nightly/",
       test(attr(deny(warnings))))]

extern crate libtest;
pub use libtest::*;
