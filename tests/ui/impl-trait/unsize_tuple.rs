//! Test that we do not allow unsizing `([Opaque; N],)` to `([Concrete],)`.

#![feature(unsized_tuple_coercion)]

fn hello() -> ([impl Sized; 2],) {
    if false {
        let x = hello();
        let _: &([i32],) = &x;
        //~^ ERROR: mismatched types
    }
    todo!()
}

fn main() {}
