//@ run-pass
//@ proc-macro: count_compound_ops.rs
//@ ignore-backends: gcc

extern crate count_compound_ops;
use count_compound_ops::count_compound_ops;

fn main() {
    assert_eq!(count_compound_ops!(foo<=>bar <<<! -baz ++), 4);
}
