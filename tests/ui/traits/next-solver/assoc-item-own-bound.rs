//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

// Regression test for rust-lang/rust#162368: an impl associated type that does
// not satisfy an explicit own clause of the trait item used to be reported as
// a projection type mismatch (E0271) by the new solver instead of the
// unsatisfied trait bound (E0277).

trait Trait {
    type Assoc
    where
        Self: Sized;
}

//[next]~v ERROR the size for values of type `T` cannot be known at compilation time
impl<T: ?Sized> Trait for T {
    type Assoc = i32;
    //[current]~^ ERROR the size for values of type `T` cannot be known at compilation time
}

fn main() {}
