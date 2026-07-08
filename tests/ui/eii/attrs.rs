#![feature(extern_item_impls)]
#![deny(deprecated)] //~ NOTE:

// makes no sense on functions, nor on the macro generated (it's a macrov2).
#[macro_export] //~ NOTE: this attribute is not supported
// makes sense, as long as we only forward it onto the function,
// so we allow and this shouln't cause errors for being on a "wrong target".
#[inline]
// makes sense, should be allowed, and forwarded on both the function and the macro
#[deprecated = "foo"]
#[eii]
fn example() {}
//~^ ERROR only a small subset of attributes are supported on externally implementable items

// check that both are deprecated vvvv
#[example]
//~^ ERROR use of deprecated macro
fn explicit_impl() {}
fn main() {
    example()
    //~^ ERROR use of deprecated function
}
