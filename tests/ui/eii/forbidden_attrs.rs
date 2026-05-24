// Tests attributes that are forbidden on EII.
#![feature(extern_item_impls)]

#[unsafe(no_mangle)]
//~^ ERROR `#[no_mangle]` cannot be used on externally implementable items
#[eii]
fn foo() {}

#[unsafe(export_name = "bar")]
//~^ ERROR `#[export_name]` cannot be used on externally implementable items
#[eii]
fn bar() {}

#[link_name = "baz"]
//~^ ERROR `#[link_name]` cannot be used on externally implementable items
#[eii]
fn baz() {}

#[eii]
fn qux();

#[unsafe(no_mangle)]
//~^ ERROR `#[no_mangle]` cannot be used on externally implementable items
#[qux]
fn qux_impl() {}

#[eii]
fn corge();

#[unsafe(export_name = "corge_impl")]
//~^ ERROR `#[export_name]` cannot be used on externally implementable items
#[corge]
fn corge_impl() {}

#[eii]
fn garply();

#[link_name = "garply_impl"]
//~^ ERROR `#[link_name]` cannot be used on externally implementable items
#[garply]
fn garply_impl() {}

fn main() {}
