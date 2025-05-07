#[unsafe(proc_macro)]
//~^ ERROR: is not an unsafe attribute
//~| ERROR attribute is only usable with crates of the `proc-macro` crate type
pub fn a() {}


#[unsafe(proc_macro_derive(Foo))]
//~^ ERROR: is not an unsafe attribute
//~| ERROR attribute is only usable with crates of the `proc-macro` crate type
pub fn b() {}

#[proc_macro_derive(unsafe(Foo))]
//~^ ERROR attribute is only usable with crates of the `proc-macro` crate type
//~| ERROR: expected identifier, found keyword `unsafe`
pub fn c() {}

#[unsafe(proc_macro_attribute)]
//~^ ERROR: is not an unsafe attribute
//~| ERROR attribute is only usable with crates of the `proc-macro` crate type
pub fn d() {}

#[unsafe(allow(dead_code))]
//~^ ERROR: is not an unsafe attribute
pub fn e() {}

#[unsafe(allow(unsafe(dead_code)))]
//~^ ERROR: is not an unsafe attribute
//~| ERROR: malformed lint attribute input
//~| ERROR: malformed lint attribute input
//~| ERROR: expected identifier, found keyword `unsafe`
//~| ERROR: malformed lint attribute input
//~| ERROR: malformed lint attribute input
//~| ERROR: malformed lint attribute input
//~| ERROR: malformed lint attribute input
pub fn f() {}

fn main() {}
