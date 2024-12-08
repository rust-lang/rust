//@ only-x86_64

#![warn(unused_attributes)]

#[target_feature(enable = "sse2")]
//~^ ERROR attribute should be applied to a function
extern crate alloc;
//~^ NOTE not a function

#[target_feature(enable = "sse2")]
//~^ ERROR attribute should be applied to a function
use alloc::alloc::alloc;
//~^ NOTE not a function

#[target_feature(enable = "sse2")]
//~^ ERROR attribute should be applied to a function
extern "Rust" {}
//~^ NOTE not a function

#[target_feature = "+sse2"]
//~^ ERROR malformed `target_feature` attribute
#[target_feature(enable = "foo")]
//~^ ERROR not valid for this target
//~| NOTE `foo` is not valid for this target
#[target_feature(bar)]
//~^ ERROR malformed `target_feature` attribute
#[target_feature(disable = "baz")]
//~^ ERROR malformed `target_feature` attribute
unsafe fn foo() {}

#[target_feature(enable = "sse2")]
//~^ ERROR `#[target_feature(..)]` can only be applied to `unsafe` functions
//~| NOTE see issue #69098
//~| NOTE: this compiler was built on YYYY-MM-DD; consider upgrading it if it is out of date
fn bar() {}
//~^ NOTE not an `unsafe` function

#[target_feature(enable = "sse2")]
//~^ ERROR attribute should be applied to a function
mod another {}
//~^ NOTE not a function

#[target_feature(enable = "sse2")]
//~^ ERROR attribute should be applied to a function
const FOO: usize = 7;
//~^ NOTE not a function

#[target_feature(enable = "sse2")]
//~^ ERROR attribute should be applied to a function
struct Foo;
//~^ NOTE not a function

#[target_feature(enable = "sse2")]
//~^ ERROR attribute should be applied to a function
enum Bar {}
//~^ NOTE not a function

#[target_feature(enable = "sse2")]
//~^ ERROR attribute should be applied to a function
union Qux {
//~^ NOTE not a function
    f1: u16,
    f2: u16,
}

#[target_feature(enable = "sse2")]
//~^ ERROR attribute should be applied to a function
type Uwu = ();
//~^ NOTE not a function

#[target_feature(enable = "sse2")]
//~^ ERROR attribute should be applied to a function
trait Baz {}
//~^ NOTE not a function

#[inline(always)]
//~^ ERROR: cannot use `#[inline(always)]`
#[target_feature(enable = "sse2")]
unsafe fn test() {}

#[target_feature(enable = "sse2")]
//~^ ERROR attribute should be applied to a function
static A: () = ();
//~^ NOTE not a function

#[target_feature(enable = "sse2")]
//~^ ERROR attribute should be applied to a function
impl Quux for u8 {}
//~^ NOTE not a function
//~| NOTE missing `foo` in implementation
//~| ERROR missing: `foo`

#[target_feature(enable = "sse2")]
//~^ ERROR attribute should be applied to a function
impl Foo {}
//~^ NOTE not a function

trait Quux {
    fn foo(); //~ NOTE `foo` from trait
}

impl Quux for Foo {
    #[target_feature(enable = "sse2")]
    //~^ ERROR `#[target_feature(..)]` can only be applied to `unsafe` functions
    //~| NOTE see issue #69098
    //~| NOTE: this compiler was built on YYYY-MM-DD; consider upgrading it if it is out of date
    fn foo() {}
    //~^ NOTE not an `unsafe` function
}

fn main() {
    #[target_feature(enable = "sse2")]
    //~^ ERROR attribute should be applied to a function
    unsafe {
        foo();
        bar();
    }
    //~^^^^ NOTE not a function

    #[target_feature(enable = "sse2")]
    //~^ ERROR attribute should be applied to a function
    || {};
    //~^ NOTE not a function
}
