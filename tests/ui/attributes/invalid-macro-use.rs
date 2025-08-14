#![deny(unused_attributes)]
//~^ NOTE the lint level is defined here

#[macro_use = 5]
//~^ ERROR valid forms for the attribute are `#[macro_use(name1, name2, ...)]` and `#[macro_use]`
extern crate std as s1;

#[macro_use(5)]
//~^ ERROR malformed `macro_use` attribute input
//~| NOTE expected a valid identifier here
//~| NOTE for more information, visit
extern crate std as s2;

#[macro_use(a = "b")]
//~^ ERROR malformed `macro_use` attribute input
//~| NOTE didn't expect any arguments here
//~| NOTE for more information, visit
extern crate std as s3;

#[macro_use(a(b))]
//~^ ERROR malformed `macro_use` attribute input
//~| NOTE didn't expect any arguments here
//~| NOTE for more information, visit
extern crate std as s4;

#[macro_use(a::b)]
//~^ ERROR malformed `macro_use` attribute input
//~| NOTE expected a valid identifier here
//~| NOTE for more information, visit
extern crate std as s5;

#[macro_use(a)]
//~^ ERROR unused attribute
#[macro_use]
//~^ NOTE attribute also specified here
extern crate std as s6;

#[macro_use]
//~^ NOTE attribute also specified here
#[macro_use(a)]
//~^ ERROR unused attribute
extern crate std as s7;

#[macro_use]
//~^ NOTE attribute also specified here
#[macro_use]
//~^ ERROR unused attribute
extern crate std as s8;

// This is fine, both are importing different names
#[macro_use(a)]
//~^ ERROR imported macro not found
#[macro_use(b)]
//~^ ERROR imported macro not found
extern crate std as s9;

fn main() {}
