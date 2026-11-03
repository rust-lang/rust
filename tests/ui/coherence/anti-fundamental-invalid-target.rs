// Test that `#[rustc_anti_fundamental]` can only be applied to traits.
// The target restriction is enforced declaratively by `ALLOWED_TARGETS`
// in the attribute parser, so applying it to a non-trait is an error.

#![feature(rustc_attrs)]

#[rustc_anti_fundamental]
//~^ ERROR attribute cannot be used on
struct NotATrait;

#[rustc_anti_fundamental]
//~^ ERROR attribute cannot be used on
fn also_not_a_trait() {}

#[rustc_anti_fundamental]
trait Ok {}

fn main() {}
