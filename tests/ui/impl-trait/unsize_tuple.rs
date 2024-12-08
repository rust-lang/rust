//! Test that we allow unsizing `([Opaque; N],)` to `([Concrete],)`.

//@check-pass

#![feature(unsized_tuple_coercion)]

fn hello() -> ([impl Sized; 2],) {
    if false {
        let x = hello();
        let _: &([i32],) = &x;
    }
    todo!()
}

fn main() {}
