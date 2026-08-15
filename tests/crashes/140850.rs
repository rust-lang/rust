//@ known-bug: #140850
//@ compile-flags: -Zvalidate-mir
fn A() -> impl {
    while A() {}
    loop {}
}
fn main() {}
