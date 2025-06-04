#![feature(closure_lifetime_binder)]

fn main() {
    let c = for<'a> |b: &'a [u32; _]| -> u32 { b[0] };
    //~^ ERROR: implicit types in closure signatures are forbidden when `for<...>` is present
    c(&[1_u32; 2]);
}
