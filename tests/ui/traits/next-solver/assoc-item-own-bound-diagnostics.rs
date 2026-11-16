//@ compile-flags: -Znext-solver

// Tests the diagnostics for associated-item own clauses (e.g. `type Assoc
// where Self: Sized;`) that are not satisfied by an impl. The recovered error
// should point at the type that fails the clause and only suggest fixes that
// would actually satisfy it.

// The impl associated type already carries other where clauses, but is
// missing the `Self: Sized` clause required by the trait item. Relaxing the
// implicit `Sized` restriction of the associated type would not satisfy that
// clause.
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

// The value of the impl associated type does not satisfy the implicit `Sized`
// restriction of the trait item. Relaxing that restriction is the right fix,
// so the suggestion must still be emitted even though the trait item carries
// other explicit where clauses.
trait TraitWithOtherClause {
    type Assoc
    where
        Self: 'static;
}

impl TraitWithOtherClause for () {
    type Assoc = [u8];
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
}

// A trait item without any explicit where clauses: the ordinary
// implicit-`Sized` diagnostic is unchanged.
trait NormalAssoc {
    type Assoc;
}

impl NormalAssoc for () {
    type Assoc = [u8];
    //~^ ERROR the size for values of type `[u8]` cannot be known at compilation time
}

// A projection use site outside of an impl: the recovery also applies there.
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

// Own clauses with traits other than `Sized` are handled the same way.
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

// Own clauses on traits with a custom `#[rustc_on_unimplemented]` diagnostic
// keep their custom label.
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
