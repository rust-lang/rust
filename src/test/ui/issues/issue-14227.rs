#![allow(safe_extern_statics, warnings)]

extern {
    pub static symbol: u32;
}
static CRASH: u32 = symbol;
//~^ ERROR could not evaluate static initializer
//~| tried to read from foreign (extern) static

fn main() {}
