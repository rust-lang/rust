//@ run-pass
//@ aux-build:same-symbol-name-for-inner-statics-issue-9188.rs

extern crate same_symbol_name_for_inner_statics_issue_9188 as lib;

pub fn main() {
    let a = lib::bar();
    let b = lib::foo::<isize>();
    assert_eq!(*a, *b);
}

// https://github.com/rust-lang/rust/issues/9188
