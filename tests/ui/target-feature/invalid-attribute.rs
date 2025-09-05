//@ only-x86_64

#![warn(unused_attributes)]

#[target_feature(enable = "sse2")]
//~^ ERROR attribute cannot be used on
extern crate alloc;

#[target_feature(enable = "sse2")]
//~^ ERROR attribute cannot be used on
use alloc::alloc::alloc;

#[target_feature(enable = "sse2")]
//~^ ERROR attribute cannot be used on
extern "Rust" {}

#[target_feature = "+sse2"]
//~^ ERROR malformed `target_feature` attribute
//~| NOTE expected this to be a list
#[target_feature(enable = "foo")]
//~^ ERROR not valid for this target
//~| NOTE `foo` is not valid for this target
#[target_feature(bar)]
//~^ ERROR malformed `target_feature` attribute
//~| NOTE expected this to be of the form `enable = "..."`
#[target_feature(disable = "baz")]
//~^ ERROR malformed `target_feature` attribute
//~| NOTE expected this to be of the form `enable = "..."`
unsafe fn foo() {}

#[target_feature(enable = "sse2")]
//~^ ERROR attribute cannot be used on
mod another {}

#[target_feature(enable = "sse2")]
//~^ ERROR attribute cannot be used on
const FOO: usize = 7;

#[target_feature(enable = "sse2")]
//~^ ERROR attribute cannot be used on
struct Foo;

#[target_feature(enable = "sse2")]
//~^ ERROR attribute cannot be used on
enum Bar {}

#[target_feature(enable = "sse2")]
//~^ ERROR attribute cannot be used on
union Qux {
        f1: u16,
    f2: u16,
}

#[target_feature(enable = "sse2")]
//~^ ERROR attribute cannot be used on
type Uwu = ();

#[target_feature(enable = "sse2")]
//~^ ERROR attribute cannot be used on
trait Baz {}

#[inline(always)]
//~^ ERROR: cannot use `#[inline(always)]`
//~| NOTE: see issue #145574 <https://github.com/rust-lang/rust/issues/145574> for more information
//~| NOTE: this compiler was built on YYYY-MM-DD; consider upgrading it if it is out of date
#[target_feature(enable = "sse2")]
unsafe fn test() {}

#[target_feature(enable = "sse2")]
//~^ ERROR attribute cannot be used on
static A: () = ();

#[target_feature(enable = "sse2")]
//~^ ERROR attribute cannot be used on
impl Quux for u8 {}
//~^ NOTE missing `foo` in implementation
//~| ERROR missing: `foo`

#[target_feature(enable = "sse2")]
//~^ ERROR attribute cannot be used on
impl Foo {}

trait Quux {
    fn foo(); //~ NOTE `foo` from trait
    //~^ NOTE: type in trait
}

impl Quux for Foo {
    #[target_feature(enable = "sse2")]
    //~^ ERROR `#[target_feature(..)]` cannot be applied to safe trait method
    //~| NOTE cannot be applied to safe trait method
    fn foo() {}
    //~^ NOTE not an `unsafe` function
    //~| ERROR: incompatible type for trait
    //~| NOTE: expected safe fn, found unsafe fn
    //~| NOTE: expected signature `fn()`
}

fn main() {
    #[target_feature(enable = "sse2")]
    //~^ ERROR attribute cannot be used on
    unsafe {
        foo();
    }

    #[target_feature(enable = "sse2")]
    //~^ ERROR attribute cannot be used on
    || {};
    }

#[target_feature(enable = "+sse2")]
//~^ ERROR `+sse2` is not valid for this target
//~| NOTE `+sse2` is not valid for this target
unsafe fn hey() {}
