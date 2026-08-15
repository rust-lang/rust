#![allow(dead_code)]

fn bar<'a>(input: &'a u32, f: fn(&'a u32) -> &'a u32) -> &'static u32 {
    // Here the NLL checker must relate the types in `f` to the types
    // in `g`. These are related via the `UnsafeFnPointer` cast.
    let g: unsafe fn(_) -> _ = f;
    unsafe { g(input) }
    //~^ ERROR lifetime may not live long enough
}

fn main() {}
