//@ known-bug: #137129
#![core::contracts::ensures]
struct A {
    b: dyn A + 'static,
}
fn c() {}
