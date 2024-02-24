//@ compile-flags: -Znext-solver

trait Setup {
    type From: Copy;
}

fn copy<U: Setup + ?Sized>(from: &U::From) -> U::From {
    *from
}

pub fn copy_any<T>(t: &T) -> T {
    copy::<dyn Setup<From=T>>(t)
    //~^ ERROR the type `&<dyn Setup<From = T> as Setup>::From` is not well-formed
    //~| ERROR the trait bound `dyn Setup<From = T>: Setup` is not satisfied
    //~| ERROR mismatched types
    //~| ERROR the type `<dyn Setup<From = T> as Setup>::From` is not well-formed

    // FIXME(-Znext-solver): These error messages are horrible and some of them
    // are even simple fallout from previous error.
}

fn main() {
    let x = String::from("Hello, world");
    let y = copy_any(&x);
    println!("{y}");
}
