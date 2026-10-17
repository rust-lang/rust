//! This is a regression test to avoid an ICE in diagnostics code.
//! A typo in the compiler used to get the DefId of FnOnce, and
//! use it where an associated item was expected.

fn frob() -> impl Fn<P, Output = T> + '_ {}
//~^ ERROR missing lifetime specifier
//~| ERROR cannot find type `P`
//~| ERROR cannot find type `T`
//~| ERROR `Fn`-family traits' type parameters is subject to change
//~| ERROR `Fn`-family traits' type parameters is subject to change

fn open_parent<'path>() {
    todo!()
}

fn main() {
    let old_path = frob("hello");
    //~^ ERROR function takes 0 arguments

    open_parent(&old_path)
    //~^ ERROR function takes 0 arguments
}
