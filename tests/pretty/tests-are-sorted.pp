#![feature(prelude_import)]
#![no_std]
#[macro_use]
extern crate std;
#[prelude_import]
use ::std::prelude::rust_2015::*;
//@ compile-flags: --crate-type=lib --test --remap-path-prefix={{src-base}}/=/the/src/ --remap-path-prefix={{src-base}}\=/the/src/
//@ pretty-compare-only
//@ pretty-mode:expanded
//@ pp-exact:tests-are-sorted.pp

extern crate test;
#[rustc_test_marker = "m_test"]
#[doc(hidden)]
pub const m_test: test::TestDescAndFn =
    test::TestDescAndFn {
        desc: test::TestDesc {
            name: test::StaticTestName("m_test"),
            ignore: false,
            ignore_message: ::core::option::Option::None,
            source_file: "/the/src/tests-are-sorted.rs",
            start_line: 7usize,
            start_col: 4usize,
            end_line: 7usize,
            end_col: 10usize,
            compile_fail: false,
            no_run: false,
            should_panic: test::ShouldPanic::No,
            test_type: test::TestType::Unknown,
        },
        testfn: test::StaticTestFn(#[coverage(off)] ||
                test::assert_test_result(m_test())),
    };
fn m_test() {}

extern crate test;
#[rustc_test_marker = "z_test"]
#[doc(hidden)]
pub const z_test: test::TestDescAndFn =
    test::TestDescAndFn {
        desc: test::TestDesc {
            name: test::StaticTestName("z_test"),
            ignore: true,
            ignore_message: ::core::option::Option::Some("not yet implemented"),
            source_file: "/the/src/tests-are-sorted.rs",
            start_line: 11usize,
            start_col: 4usize,
            end_line: 11usize,
            end_col: 10usize,
            compile_fail: false,
            no_run: false,
            should_panic: test::ShouldPanic::No,
            test_type: test::TestType::Unknown,
        },
        testfn: test::StaticTestFn(#[coverage(off)] ||
                test::assert_test_result(z_test())),
    };
#[ignore = "not yet implemented"]
fn z_test() {}

extern crate test;
#[rustc_test_marker = "a_test"]
#[doc(hidden)]
pub const a_test: test::TestDescAndFn =
    test::TestDescAndFn {
        desc: test::TestDesc {
            name: test::StaticTestName("a_test"),
            ignore: false,
            ignore_message: ::core::option::Option::None,
            source_file: "/the/src/tests-are-sorted.rs",
            start_line: 14usize,
            start_col: 4usize,
            end_line: 14usize,
            end_col: 10usize,
            compile_fail: false,
            no_run: false,
            should_panic: test::ShouldPanic::No,
            test_type: test::TestType::Unknown,
        },
        testfn: test::StaticTestFn(#[coverage(off)] ||
                test::assert_test_result(a_test())),
    };
fn a_test() {}
#[rustc_main]
#[coverage(off)]
#[doc(hidden)]
pub fn main() -> () {
    extern crate test;
    test::test_main_static(&[&a_test, &m_test, &z_test])
}
