// Test that `#[link_args]` attribute is gated by `link_args`
// feature gate, both when it occurs where expected (atop
// `extern { }` blocks) and where unexpected.

// sidestep warning (which is correct, but misleading for
// purposes of this test)
#![allow(unused_attributes)]

#![link_args = "-l unexpected_use_as_inner_attr_on_mod"]
//~^ ERROR the `link_args` attribute is experimental

#[link_args = "-l expected_use_case"]
//~^ ERROR the `link_args` attribute is experimental
extern {}

#[link_args = "-l unexected_use_on_non_extern_item"]
//~^ ERROR: the `link_args` attribute is experimental
fn main() {}
