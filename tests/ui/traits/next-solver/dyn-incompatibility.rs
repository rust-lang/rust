//@ compile-flags: -Znext-solver

trait Setup {
    type From: Copy;
}

fn copy<U: Setup + ?Sized>(from: &U::From) -> U::From {
    *from
}

pub fn copy_any<T>(t: &T) -> T {
    copy::<dyn Setup<From=T>>(t)
    //~^ ERROR the trait bound `T: Copy` is not satisfied in `dyn Setup<From = T>`
    //~| ERROR mismatched types
    //~| ERROR the trait bound `T: Copy` is not satisfied

    // FIXME(-Znext-solver): These error messages are horrible and some of them
    // are even simple fallout from previous error.
}

fn main() {
    let x = String::from("Hello, world");
    let y = copy_any(&x);
    println!("{y}");
}
