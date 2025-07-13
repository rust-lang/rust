//@ run-pass
//@ aux-build:reexported-structs-impls-link-error-issue-9906.rs


extern crate reexported_structs_impls_link_error_issue_9906 as testmod;

pub fn main() {
    testmod::foo();
    testmod::FooBar::new(1);
}

// https://github.com/rust-lang/rust/issues/9906
