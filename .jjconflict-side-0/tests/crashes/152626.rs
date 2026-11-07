//@ known-bug: #152626
//@ needs-rustc-debug-assertions
struct A<T: Into<u32>>(T);
fn f() -> A<&'static ()> {
    todo!()
}
fn main() {}
