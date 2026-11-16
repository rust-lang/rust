// This should fail a normal compile due to non_camel_case_types,
// It should pass a doc-compile as it only needs to type-check and
// therefore should not concern itself with the lints.

//@ check-pass

#[deny(warnings)]

pub struct Foo {
    field: i32,
}
