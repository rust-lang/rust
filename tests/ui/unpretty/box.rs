//@ compile-flags: -Zunpretty=thir-tree
//@ check-pass

#![feature(liballoc_internals)]

fn main() {
    let _ = std::boxed::box_new(1);
}
