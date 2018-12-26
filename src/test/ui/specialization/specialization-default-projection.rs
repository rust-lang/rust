#![feature(specialization)]

// Make sure we can't project defaulted associated types

trait Foo {
    type Assoc;
}

impl<T> Foo for T {
    default type Assoc = ();
}

impl Foo for u8 {
    type Assoc = String;
}

fn generic<T>() -> <T as Foo>::Assoc {
    // `T` could be some downstream crate type that specializes (or,
    // for that matter, `u8`).

    () //~ ERROR mismatched types
}

fn monomorphic() -> () {
    // Even though we know that `()` is not specialized in a
    // downstream crate, typeck refuses to project here.

    generic::<()>() //~ ERROR mismatched types
}

fn main() {
    // No error here, we CAN project from `u8`, as there is no `default`
    // in that impl.
    let s: String = generic::<u8>();
    println!("{}", s); // bad news if this all compiles
}
