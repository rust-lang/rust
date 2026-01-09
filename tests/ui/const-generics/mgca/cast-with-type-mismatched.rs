//! regression test for <https://github.com/rust-lang/rust/issues/143506>
#![expect(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(min_generic_const_args)]

fn foo<const N: u32>(a: [(); N as usize]) {}
//~^ ERROR: complex const arguments must be placed inside of a `const` block

const C: f32 = 1.0;

fn main() {
    foo::<C>([]);
    //~^ ERROR: the constant `C` is not of type `u32`
}
