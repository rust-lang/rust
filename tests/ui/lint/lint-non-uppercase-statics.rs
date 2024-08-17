#![forbid(non_upper_case_globals)]
#![allow(dead_code)]

static foo1: isize = 1; //~ ERROR static variable `foo1` should have an upper case name

static mut foo2: isize = 1; //~ ERROR static variable `foo2` should have an upper case name

#[no_mangle]
pub static extern_foo: isize = 1; // OK, because #[no_mangle] supersedes the warning

static BAR_ONE: isize = 1;
use BAR_ONE as Bar1; //~ ERROR renamed static variable `Bar1` should have an upper case name

static mut BAR_TWO: isize = 1;
use BAR_TWO as Bar2; //~ ERROR renamed static variable `Bar2` should have an upper case name

fn main() {}
