//! This test checks that even if some associated types have
//! `where Self: Sized` bounds, those without still need to be
//! mentioned in trait objects.

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
    //~^ ERROR: the size for values of type `dyn Bop` cannot be known at compilation time
}
