//! Regression test for <https://github.com/rust-lang/rust/issues/27815>.
//! Test usage of struct literal syntax with non-structs doesn't ICE.

mod A {}

fn main() {
    let u = A { x: 1 }; //~ ERROR expected struct, variant or union type, found module `A`
    let v = u32 { x: 1 }; //~ ERROR expected struct, variant or union type, found builtin type `u32`
    match () {
        A { x: 1 } => {}
        //~^ ERROR expected struct, variant or union type, found module `A`
        u32 { x: 1 } => {}
        //~^ ERROR expected struct, variant or union type, found builtin type `u32`
    }
}
