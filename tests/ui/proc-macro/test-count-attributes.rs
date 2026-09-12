//@ check-pass
//@ proc-macro: test-count-attributes.rs
//@ compile-flags: --test
// Tests whether attributes on tests can be observed by proc macros
// Regression test for https://github.com/rust-lang/rust/issues/161920

#[test]
#[test_count_attributes::assert_no_attributes]
fn meow1() {}

#[test_count_attributes::assert_one_attribute]
#[test]
fn meow2() {}

#[test]
#[should_panic]
#[test_count_attributes::assert_one_attribute]
fn meow3() {}

#[test]
#[test_count_attributes::assert_one_attribute]
#[should_panic]
fn meow4() {}

#[test_count_attributes::assert_two_attributes]
#[test]
#[should_panic]
fn meow5() {}

fn main() {}
