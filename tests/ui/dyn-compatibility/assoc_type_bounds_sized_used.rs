//! This test checks that associated types with `Self: Sized` cannot be projected
//! from a `dyn Trait`.

trait Bop {
    type Bar: Default
    where
        Self: Sized;
}

fn bop<T: Bop + ?Sized>() {
    let _ = <T as Bop>::Bar::default();
    //~^ ERROR: trait bounds were not satisfied
    //~| ERROR: the size for values of type `T` cannot be known at compilation time
}

fn main() {
    bop::<dyn Bop>();
}
