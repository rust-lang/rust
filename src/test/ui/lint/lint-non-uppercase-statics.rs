#![forbid(non_upper_case_globals)]
#![allow(dead_code)]

static foo: isize = 1; //~ ERROR static variable `foo` should have an upper case name such as `FOO`

static mut bar: isize = 1;
        //~^ ERROR static variable `bar` should have an upper case name such as `BAR`

#[no_mangle]
pub static extern_foo: isize = 1; // OK, because #[no_mangle] supersedes the warning

fn main() { }
