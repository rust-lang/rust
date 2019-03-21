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
#![feature(staged_api)]
#![feature(test)]
#![feature(rustc_private)]

extern crate libtest;

// FIXME: we should be more explicit about the exact APIs that we
// export to users.
pub use libtest::{
    assert_test_result, filter_tests, parse_opts, run_test, test_main, test_main_static,
    Bencher, TestFn::DynTestFn, TestName::DynTestName, Metric, MetricMap, Options,
    RunIgnored, ShouldPanic, TestFn::StaticBenchFn, TestFn::StaticTestFn,
    TestName::StaticTestName, TestDesc, TestDescAndFn, TestName, TestOpts,
    TestResult, TestResult::TrFailed, TestResult::TrFailedMsg,
    TestResult::TrIgnored, TestResult::TrOk, stats::Summary
};

pub use std::hint::black_box;

#[cfg(test)]
mod tests {
    use crate::Bencher;
    use libtest::stats::Stats;

    #[bench]
    pub fn sum_three_items(b: &mut Bencher) {
        b.iter(|| {
            [1e20f64, 1.5f64, -1e20f64].sum();
        })
    }

    #[bench]
    pub fn sum_many_f64(b: &mut Bencher) {
        let nums = [-1e30f64, 1e60, 1e30, 1.0, -1e60];
        let v = (0..500).map(|i| nums[i % 5]).collect::<Vec<_>>();
        b.iter(|| {
            v.sum();
        })
    }

    #[bench]
    pub fn no_iter(_: &mut Bencher) {}
}
