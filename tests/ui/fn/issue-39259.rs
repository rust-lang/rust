#![feature(fn_traits)]
#![feature(unboxed_closures)]

struct S;

impl Fn(u32) -> u32 for S {
//~^ ERROR associated type bindings are not allowed here [E0229]
    fn call(&self) -> u32 {
        5
    }
}

fn main() {}
