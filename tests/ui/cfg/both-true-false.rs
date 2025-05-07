/// Test that placing a `cfg(true)` and `cfg(false)` on the same item result in
//. it being disabled.`

#[cfg(false)]
#[cfg(true)]
fn foo() {}

#[cfg(true)]
#[cfg(false)]
fn foo() {}

fn main() {
    foo();  //~ ERROR cannot find function `foo` in this scope
}
