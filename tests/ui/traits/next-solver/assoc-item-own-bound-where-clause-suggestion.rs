//@ compile-flags: -Znext-solver

// Case B: implementation associated type already has an existing where-clause
trait TraitWithBound {
    type Assoc
    where
        Self: Sized,
        Self: 'static;
}

impl<T: ?Sized> TraitWithBound for T {
    //~^ ERROR the size for values of type `T` cannot be known at compilation time
    type Assoc = i32
    where
        Self: 'static;
}

// Case C: normal associated-type implicit-Sized diagnostic (suppression is not global)
trait NormalAssoc {
    type Assoc;
}

impl NormalAssoc for () {
    type Assoc = [u8];
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
}

// Case D: projection/non-impl use-site (targeted suggestion does not leak)
trait NonImplTrait {
    type Assoc
    where
        Self: Sized;
}

fn non_impl_use<T: ?Sized>() {
    let _: <T as NonImplTrait>::Assoc;
    //~^ ERROR the size for values of type `T` cannot be known at compilation time
    //~| ERROR the trait bound `T: NonImplTrait` is not satisfied
}

// Case E: explicit associated-type bound with a trait other than `Sized`
trait Marker {}

trait TraitWithMarker {
    type Assoc
    where
        Self: Marker;
}

impl<T> TraitWithMarker for T {
    //~^ ERROR the trait bound `T: Marker` is not satisfied
    type Assoc = i32;
}

// Case F: associated-type bound with a custom on-unimplemented diagnostic
trait TraitWithSend {
    type Assoc
    where
        Self: Send;
}

impl<T> TraitWithSend for T {
    //~^ ERROR `T` cannot be sent between threads safely
    type Assoc = i32;
}

fn main() {}
