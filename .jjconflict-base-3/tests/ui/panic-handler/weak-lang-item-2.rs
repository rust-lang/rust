//@ run-pass
//@ aux-build:weak-lang-items.rs


extern crate weak_lang_items as other;

fn main() {
    // The goal of the test is just to make sure other::foo() is referenced at link time. Since
    // the function panics, to prevent it from running we gate it behind an always-false `if` that
    // is not going to be optimized away.
    if std::hint::black_box(false) {
        other::foo();
    }
}
