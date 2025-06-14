#![feature(closure_lifetime_binder)]

struct Foo<const N: usize>([u32; N]);

fn main() {
    let c = for<'a> |b: &'a Foo<_>| -> u32 { b.0[0] };
    //~^ ERROR: implicit types in closure signatures are forbidden when `for<...>` is present
    c(&Foo([1_u32; 1]));
}
