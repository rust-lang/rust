// Regression test for #152335.
// The compiler used to ICE in `generics_of` when `note_and_explain_type_err`
// was called with `CRATE_DEF_ID` as the body owner during dyn-compatibility
// checking. This happened because `ObligationCause::dummy()` sets `body_id`
// to `CRATE_DEF_ID`, and error reporting tried to look up generics on it.

struct ActuallySuper;

trait Super<B = Self> {
    type Assoc;
}

trait Dyn {}
impl<T, U> Dyn for dyn Foo<T, U> + '_ {}
//~^ ERROR the trait `Foo` is not dyn compatible

trait Foo<T, U>: Super<ActuallySuper, Assoc = T>
//~^ ERROR type mismatch resolving
//~| ERROR the size for values of type `Self` cannot be known
where
    <Self as Mirror>::Assoc: Super,
    //~^ ERROR type mismatch resolving
    //~| ERROR the size for values of type `Self` cannot be known
    //~| ERROR type mismatch resolving
    //~| ERROR the size for values of type `Self` cannot be known
{
    fn transmute(&self, t: T) -> <Self as B>::Assoc;
    //~^ ERROR cannot find trait `B` in this scope
    //~| ERROR type mismatch resolving
    //~| ERROR the size for values of type `Self` cannot be known
    //~| ERROR type mismatch resolving
    //~| ERROR the size for values of type `Self` cannot be known
}

trait Mirror {
    type Assoc: ?Sized;
}
impl<T: Super<ActuallySuper, Assoc = T>> Mirror for T {}
//~^ ERROR not all trait items implemented
//~| ERROR `main` function not found
