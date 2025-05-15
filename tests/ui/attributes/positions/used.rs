//! Checks that `#[used]` cannot be used on invalid positions.
#![crate_type = "lib"]

#[used]
static FOO: u32 = 0; // OK

#[used] //~ ERROR attribute must be applied to a `static` variable
fn foo() {}

#[used] //~ ERROR attribute must be applied to a `static` variable
struct Foo {}

#[used] //~ ERROR attribute must be applied to a `static` variable
trait Bar {}

#[used] //~ ERROR attribute must be applied to a `static` variable
impl Bar for Foo {}

// Regression test for <https://github.com/rust-lang/rust/issues/126789>.
extern "C" {
    #[used] //~ ERROR attribute must be applied to a `static` variable
    static BAR: i32;
}
