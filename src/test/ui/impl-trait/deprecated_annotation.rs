// compile-pass

#![deny(warnings)]

#[deprecated]
trait Deprecated {}

#[deprecated]
struct DeprecatedTy;

#[allow(deprecated)]
impl Deprecated for DeprecatedTy {}

#[allow(deprecated)]
fn foo() -> impl Deprecated { DeprecatedTy }

fn main() {
    foo();
}
