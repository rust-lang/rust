//! Checks that `#[used]` cannot be used on invalid positions.
#![crate_type = "lib"]

#[used]
static FOO: u32 = 0; // OK

#[used] //~ ERROR attribute cannot be used on
fn foo() {}

#[used] //~ ERROR attribute cannot be used on
struct Foo {}

#[used] //~ ERROR attribute cannot be used on
trait Bar {}

#[used] //~ ERROR attribute cannot be used on
impl Bar for Foo {}

// Regression test for <https://github.com/rust-lang/rust/issues/126789>.
extern "C" {
    #[used] //~ ERROR attribute cannot be used on
    static BAR: i32;
}
