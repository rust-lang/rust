#![feature(const_generics)]
#![feature(const_generics_defaults)]
#![allow(incomplete_features)]

pub struct Example<const N: usize=13>;
pub struct Example2<T=u32, const N: usize=13>(T);
pub struct Example3<const N: usize=13, T=u32>(T);
pub struct Example4<const N: usize=13, const M: usize=4>;

fn main() {
    let e: Example::<13> = ();
    //~^ Error: mismatched types
    let e: Example2::<u32, 13> = ();
    //~^ Error: mismatched types
    let e: Example3::<13, u32> = ();
    //~^ Error: mismatched types
    let e: Example3::<7> = ();
    //~^ Error: mismatched types
    // FIXME(const_generics_defaults): There should be a note for the error below, but it is
    // missing.
    let e: Example4::<7> = ();
    //~^ Error: mismatched types
}
