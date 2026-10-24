//! Check that we do not suggest using `#[derive(...)]` for unions,
//! as some traits cannot be autoderived for them.
//@ dont-require-annotations: NOTE

union U { //~ HELP consider annotating `U` with `#[derive(Clone)]`
         //~| HELP consider annotating `U` with `#[derive(Copy)]`
         //~| HELP the trait `Debug` is not implemented for `U`
         //~| HELP the trait `Default` is not implemented for `U`
         //~| HELP the trait `Hash` is not implemented for `U`
    a: u8,
}

fn x<T: Clone>() {}
fn y<T: Copy>() {}

fn main() {
    let u = U { a: 0 };
    // Debug
    println!("{u:?}"); //~ ERROR `U` doesn't implement `Debug`
                       //~| NOTE manually `impl Debug for U`
    // PartialEq
    let _ = u == U { a: 0 }; //~ ERROR binary operation `==` cannot be applied to type `U`
                              //~| NOTE the trait `PartialEq` must be implemented
    // PartialOrd
    let _ = u < U { a: 1 }; //~ ERROR binary operation `<` cannot be applied to type `U`
                             //~| NOTE the trait `PartialOrd` must be implemented
    // Default
    let _: U = Default::default(); //~ ERROR the trait bound `U: Default` is not satisfied
    // Hash
    let mut h = std::collections::hash_map::DefaultHasher::new();
    std::hash::Hash::hash(&u, &mut h); //~ ERROR the trait bound `U: Hash` is not satisfied

    // Clone
    x::<U>(); //~ ERROR the trait bound `U: Clone` is not satisfied
    // Copy
    y::<U>(); //~ ERROR the trait bound `U: Copy` is not satisfied
}
