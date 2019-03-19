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
#![feature(asm)]
#![feature(staged_api)]
#![feature(test)]

extern crate libtest;

// FIXME: we should be more explicit about the exact APIs that we
// export to users.
pub use libtest::{
    assert_test_result, filter_tests, parse_opts, run_test, test_main, test_main_static,
    Bencher, DynTestFn, DynTestName, Metric, MetricMap, Options, RunIgnored, ShouldPanic,
    StaticBenchFn, StaticTestFn, StaticTestName, TestDesc, TestDescAndFn, TestName, TestOpts,
    TestResult, TrFailed, TrFailedMsg, TrIgnored, TrOk, stats::Summary
};

/// A function that is opaque to the optimizer, to allow benchmarks to
/// pretend to use outputs to assist in avoiding dead-code
/// elimination.
///
/// This function is a no-op, and does not even read from `dummy`.
#[cfg(not(any(target_arch = "asmjs", target_arch = "wasm32")))]
pub fn black_box<T>(dummy: T) -> T {
    // we need to "use" the argument in some way LLVM can't
    // introspect.
    unsafe { asm!("" : : "r"(&dummy)) }
    dummy
}
#[cfg(any(target_arch = "asmjs", target_arch = "wasm32"))]
#[inline(never)]
pub fn black_box<T>(dummy: T) -> T {
    dummy
}
