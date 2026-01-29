// This test verfies that `#[export_visibility = ...]` will report an error
// when the argument cannot be parsed.
#![feature(export_visibility)]
#[no_mangle]
#[export_visibility = "unrecognized visibility value"]
//~^ ERROR: invalid export visibility
pub static TESTED_STATIC: [u8; 6] = *b"foobar";

// The following `static`s verify that `hidden`, `protected`, and `interposable`
// are not supported yet.
#[no_mangle]
#[export_visibility = "hidden"]
//~^ ERROR: invalid export visibility
pub static TESTED_STATIC_HIDDEN: [u8; 6] = *b"foobar";
#[no_mangle]
#[export_visibility = "protected"]
//~^ ERROR: invalid export visibility
pub static TESTED_STATIC_PROTECTED: [u8; 6] = *b"foobar";
#[no_mangle]
#[export_visibility = "interposable"]
//~^ ERROR: invalid export visibility
pub static TESTED_STATIC_INTERPOSABLE: [u8; 6] = *b"foobar";

// The following `static`s verify that other visibility spellings are also not supported.
#[no_mangle]
#[export_visibility = "default"]
//~^ ERROR: invalid export visibility
pub static TESTED_STATIC_DEFAULT: [u8; 6] = *b"foobar";
#[no_mangle]
#[export_visibility = "public"]
//~^ ERROR: invalid export visibility
pub static TESTED_STATIC_PUBLIC: [u8; 6] = *b"foobar";

fn main() {}
