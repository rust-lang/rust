// Make sure we don't ICE printing `impl AsyncFnOnce<()>`.

#![feature(unboxed_closures, fn_traits)]

fn f() -> impl FnOnce<()> { || () }

fn main() { () = f(); }
//~^ ERROR mismatched types
