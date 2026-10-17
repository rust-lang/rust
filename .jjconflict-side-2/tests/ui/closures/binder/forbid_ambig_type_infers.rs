#![feature(closure_lifetime_binder)]

struct Foo<T>(T);

fn main() {
    let c = for<'a> |b: &'a Foo<_>| -> u32 { b.0 };
    //~^ ERROR: implicit types in closure signatures are forbidden when `for<...>` is present
    c(&Foo(1_u32));
}
