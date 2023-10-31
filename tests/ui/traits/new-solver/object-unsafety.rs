// compile-flags: -Ztrait-solver=next

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
    //~| ERROR mismatched types
    //~| ERROR the type `<dyn Setup<From = T> as Setup>::From` is not well-formed
    //~| ERROR the size for values of type `<dyn Setup<From = T> as Setup>::From` cannot be known at compilation time

    // FIXME(-Ztrait-solver=next): These error messages are horrible and some of them
    // are even simple fallout from previous error.
}

fn main() {
    let x = String::from("Hello, world");
    let y = copy_any(&x);
    println!("{y}");
}
