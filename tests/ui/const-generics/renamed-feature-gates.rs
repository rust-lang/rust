#![feature(min_generic_const_args)]
//~^ ERROR feature has been removed
//~| NOTE feature has been removed
//~| NOTE removed in
//~| NOTE renamed to `gca_min_const_items`
#![feature(generic_const_args)]
//~^ ERROR feature has been removed
//~| NOTE feature has been removed
//~| NOTE removed in
//~| NOTE renamed to `gca_const_items`
#![feature(macroless_generic_const_args)]
//~^ ERROR feature has been removed
//~| NOTE feature has been removed
//~| NOTE removed in
//~| NOTE renamed to `gca_macroless_args`
#![feature(macroless_const_item_generic_const_args)]
//~^ ERROR feature has been removed
//~| NOTE feature has been removed
//~| NOTE removed in
//~| NOTE renamed to `gca_macroless_items`
fn main() {}
