#[cfg(panic = "unwind")]
//~^ ERROR `cfg(panic)` is experimental and subject to change
fn foo() -> bool { true }
#[cfg(not(panic = "unwind"))]
//~^ ERROR `cfg(panic)` is experimental and subject to change
fn foo() -> bool { false }


fn main() {
    assert!(foo());
}
