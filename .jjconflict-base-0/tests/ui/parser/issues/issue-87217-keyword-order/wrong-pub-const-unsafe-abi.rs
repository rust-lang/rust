// Test for issue https://github.com/rust-lang/rust/issues/140171
// There is an order to respect for keywords before a function:
// `<visibility>, const, async, unsafe, extern, "<ABI>"`
//
// This test ensures the compiler is helpful about them being misplaced.
//@ edition:2018

unsafe extern "C" { static errno: i32; }

extern "C" pub const unsafe fn c() {}
//~^ ERROR expected `fn`, found keyword `pub`
//~| NOTE expected `fn`
//~| HELP visibility `pub` must come before `extern "C"`
//~| SUGGESTION pub extern "C"

fn main() {}
