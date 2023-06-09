#[doc(inline)]
//~^ ERROR conflicting
#[doc(no_inline)]
pub extern crate core;

// no warning
pub extern crate alloc;

fn main() {}
