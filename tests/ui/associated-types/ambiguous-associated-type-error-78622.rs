// https://github.com/rust-lang/rust/issues/78622
#![crate_type = "lib"]

struct S;
fn f() {
    S::A::<f> {}
    //~^ ERROR ambiguous associated type
}
