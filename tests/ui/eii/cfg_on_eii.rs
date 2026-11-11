//@ compile-flags: -Z unpretty=hir,typed

#![feature(extern_item_impls)]
#![deny(deprecated)] //~ NOTE:

#[cfg(true)]
#[eii]
fn cfg_on_eii() {}

#[cfg_attr(true, eii)]
fn conditional_eii() {}

#[cfg_attr(true, deprecated = "bar")]
#[eii]
fn cfg_attr_on_eii() {}

fn main() {
    cfg_attr_on_eii();
    //~^ ERROR use of deprecated function
}
