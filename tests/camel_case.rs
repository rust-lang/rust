extern crate clippy_lints;

use clippy_lints::utils::{camel_case_from, camel_case_until};

#[test]
fn from_full() {
    assert_eq!(camel_case_from("AbcDef"), 0);
    assert_eq!(camel_case_from("Abc"), 0);
}

#[test]
fn from_partial() {
    assert_eq!(camel_case_from("abcDef"), 3);
    assert_eq!(camel_case_from("aDbc"), 1);
}

#[test]
fn from_not() {
    assert_eq!(camel_case_from("AbcDef_"), 7);
    assert_eq!(camel_case_from("AbcDD"), 5);
}

#[test]
fn from_caps() {
    assert_eq!(camel_case_from("ABCD"), 4);
}

#[test]
fn until_full() {
    assert_eq!(camel_case_until("AbcDef"), 6);
    assert_eq!(camel_case_until("Abc"), 3);
}

#[test]
fn until_not() {
    assert_eq!(camel_case_until("abcDef"), 0);
    assert_eq!(camel_case_until("aDbc"), 0);
}

#[test]
fn until_partial() {
    assert_eq!(camel_case_until("AbcDef_"), 6);
    assert_eq!(camel_case_until("CallTypeC"), 8);
    assert_eq!(camel_case_until("AbcDD"), 3);
}

#[test]
fn until_caps() {
    assert_eq!(camel_case_until("ABCD"), 0);
}
