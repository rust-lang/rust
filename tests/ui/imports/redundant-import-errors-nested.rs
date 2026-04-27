// Issue #153156: Too many errors for missing crate for nested imports and later uses

use foo::{bar, baz::bat};
//~^ ERROR unresolved import `foo`

pub fn main() {
    foo::qux();
}
