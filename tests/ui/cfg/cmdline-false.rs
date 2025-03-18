/// Test that `--cfg false` doesn't cause `cfg(false)` to evaluate to `true`
//@ compile-flags: --cfg false

#[cfg(false)]
fn foo() {}

fn main() {
    foo();  //~ ERROR cannot find function `foo` in this scope
}
