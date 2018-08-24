#![allow(safe_extern_statics, warnings)]

extern {
    pub static symbol: ();
}
static CRASH: () = symbol;
//~^ ERROR could not evaluate static initializer
//~| tried to read from foreign (extern) static

fn main() {}
