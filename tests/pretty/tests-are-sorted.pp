#![feature(prelude_import)]
#![no_std]
#[prelude_import]
use ::std::prelude::rust_2015::*;
#[macro_use]
extern crate std;
// compile-flags: --crate-type=lib --test
// pretty-compare-only
// pretty-mode:expanded
// pp-exact:tests-are-sorted.pp

extern crate test;
#[cfg(test)]
#[rustc_test_marker = "m_test"]
pub const m_test: test::TestDescAndFn =
    test::TestDescAndFn {
        desc: test::TestDesc {
            name: test::StaticTestName("m_test"),
            ignore: false,
            ignore_message: ::core::option::Option::None,
            location_info: "/checkout/tests/pretty/tests-are-sorted.rs:7:4: 7:10",
            compile_fail: false,
            no_run: false,
            should_panic: test::ShouldPanic::No,
            test_type: test::TestType::Unknown,
        },
        testfn: test::StaticTestFn(|| test::assert_test_result(m_test())),
    };
fn m_test() {}

extern crate test;
#[cfg(test)]
#[rustc_test_marker = "z_test"]
pub const z_test: test::TestDescAndFn =
    test::TestDescAndFn {
        desc: test::TestDesc {
            name: test::StaticTestName("z_test"),
            ignore: false,
            ignore_message: ::core::option::Option::None,
            location_info: "/checkout/tests/pretty/tests-are-sorted.rs:10:4: 10:10",
            compile_fail: false,
            no_run: false,
            should_panic: test::ShouldPanic::No,
            test_type: test::TestType::Unknown,
        },
        testfn: test::StaticTestFn(|| test::assert_test_result(z_test())),
    };
fn z_test() {}

extern crate test;
#[cfg(test)]
#[rustc_test_marker = "a_test"]
pub const a_test: test::TestDescAndFn =
    test::TestDescAndFn {
        desc: test::TestDesc {
            name: test::StaticTestName("a_test"),
            ignore: false,
            ignore_message: ::core::option::Option::None,
            location_info: "/checkout/tests/pretty/tests-are-sorted.rs:13:4: 13:10",
            compile_fail: false,
            no_run: false,
            should_panic: test::ShouldPanic::No,
            test_type: test::TestType::Unknown,
        },
        testfn: test::StaticTestFn(|| test::assert_test_result(a_test())),
    };
fn a_test() {}
#[rustc_main]
pub fn main() -> () {
    extern crate test;
    test::test_main_static(&[&a_test, &m_test, &z_test])
}
