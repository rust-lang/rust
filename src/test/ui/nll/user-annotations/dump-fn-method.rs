// Unit test for the "user substitutions" that are annotated on each
// node.

// compile-flags:-Zverbose

#![feature(nll)]
#![feature(rustc_attrs)]

// Note: we reference the names T and U in the comments below.
trait Bazoom<T> {
    fn method<U>(&self, arg: T, arg2: U) { }
}

impl<S, T> Bazoom<T> for S {
}

fn foo<'a, T>(_: T) { }

#[rustc_dump_user_substs]
fn main() {
    // Here: nothing is given, so we don't have any annotation.
    let x = foo;
    x(22);

    // Here: `u32` is given, which doesn't contain any lifetimes, so we don't
    // have any annotation.
    let x = foo::<u32>;
    x(22);

    let x = foo::<&'static u32>; //~ ERROR [&ReStatic u32]
    x(&22);

    // Here: we only want the `T` to be given, the rest should be variables.
    //
    // (`T` refers to the declaration of `Bazoom`)
    let x = <_ as Bazoom<u32>>::method::<_>; //~ ERROR [^0, u32, ^1]
    x(&22, 44, 66);

    // Here: all are given and definitely contain no lifetimes, so we
    // don't have any annotation.
    let x = <u8 as Bazoom<u16>>::method::<u32>;
    x(&22, 44, 66);

    // Here: all are given and we have a lifetime.
    let x = <u8 as Bazoom<&'static u16>>::method::<u32>; //~ ERROR [u8, &ReStatic u16, u32]
    x(&22, &44, 66);

    // Here: we want in particular that *only* the method `U`
    // annotation is given, the rest are variables.
    //
    // (`U` refers to the declaration of `Bazoom`)
    let y = 22_u32;
    y.method::<u32>(44, 66); //~ ERROR [^0, ^1, u32]

    // Here: nothing is given, so we don't have any annotation.
    let y = 22_u32;
    y.method(44, 66);
}
