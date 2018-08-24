// aux-build:count_compound_ops.rs
// ignore-stage1

#![feature(proc_macro_non_items)]

extern crate count_compound_ops;
use count_compound_ops::count_compound_ops;

fn main() {
    assert_eq!(count_compound_ops!(foo<=>bar <<<! -baz ++), 4);
}
