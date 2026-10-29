#![allow(dead_code, mismatched_lifetime_syntaxes)]

use std::error::Error;

fn explicit_bound<'a, T: Error + 'static>(err: &(dyn Error + 'a)) -> bool {
    err.is::<T>() //~ ERROR borrowed data escapes outside of function
}

fn explicit_placeholder<T: Error + 'static>(err: &(dyn Error + '_)) -> bool {
    err.is::<T>() //~ ERROR borrowed data escapes outside of function
}

fn return_static(err: &dyn Error) -> &'static dyn Error {
    err //~ ERROR lifetime may not live long enough
}

fn return_other<'a, 'b>(err: &'a dyn Error) -> &'b dyn Error {
    err //~ ERROR lifetime may not live long enough
}

fn takes_static_reference(_: &'static dyn Error) {}

fn static_reference_argument(err: &dyn Error) {
    takes_static_reference(err); //~ ERROR borrowed data escapes outside of function
}

fn takes_static_object_and_reference(_: &'static (dyn Error + 'static)) {}

fn static_reference_and_object(err: &dyn Error) {
    takes_static_object_and_reference(err); //~ ERROR borrowed data escapes outside of function
}

fn takes_object_and_static_reference(_: &(dyn Error + 'static), _: &'static dyn Error) {}

fn repeated_static_reference(err: &dyn Error) {
    takes_object_and_static_reference(err, err); //~ ERROR borrowed data escapes outside of function
}

fn takes_object_and_reference<'a>(_: &(dyn Error + 'a), _: &'static u32) {}

fn unrelated_parameter<'a>(err: &'a dyn Error, other: &'a u32) {
    takes_object_and_reference(err, other); //~ ERROR borrowed data escapes outside of function
}

fn takes_generic_static<T: ?Sized + 'static>(_: &T) {}

fn generic_bound(err: &dyn Error) {
    takes_generic_static(err); //~ ERROR borrowed data escapes outside of function
}

trait Trait {
    fn needs_static(&self) where Self: 'static {}
}

fn where_self(err: &dyn Trait) {
    err.needs_static(); //~ ERROR borrowed data escapes outside of function
}

fn elided_output<T: Error + 'static>(err: &dyn Error) -> Option<&T> {
    err.downcast_ref::<T>() //~ ERROR borrowed data escapes outside of function
}

fn named_input_elided_output<'a, T: Error + 'static>(err: &'a dyn Error) -> Option<&T> {
    err.downcast_ref::<T>() //~ ERROR borrowed data escapes outside of function
}

fn placeholder_output<T: Error + 'static>(err: &dyn Error) -> Option<&'_ T> {
    err.downcast_ref::<T>() //~ ERROR borrowed data escapes outside of function
}

fn takes_predicate_reference<'a: 'static>(_: &'a (dyn Error + 'static)) {}

fn predicate_reference(err: &dyn Error) {
    takes_predicate_reference(err); //~ ERROR borrowed data escapes outside of function
}

fn takes_transitive_reference<'a: 'b, 'b: 'static>(_: &'a (dyn Error + 'static)) {}

fn transitive_reference(err: &dyn Error) {
    takes_transitive_reference(err); //~ ERROR borrowed data escapes outside of function
}

fn takes_reference_type<'a>(_: &'a (dyn Error + 'static))
where
    &'a (dyn Error + 'static): 'static,
{
}

fn reference_type_bound(err: &dyn Error) {
    takes_reference_type(err); //~ ERROR borrowed data escapes outside of function
}

trait LifetimeArgument<'a> {}

fn takes_static_argument(_: &(dyn LifetimeArgument<'static> + 'static)) {}

fn mismatched_argument<'a>(arg: &'a dyn LifetimeArgument<'a>) {
    takes_static_argument(arg); //~ ERROR borrowed data escapes outside of function
}

trait AssociatedType {
    type Item;
}

fn takes_static_associated_type(_: &(dyn AssociatedType<Item = &'static u32> + 'static)) {}

fn mismatched_associated_type<'a>(arg: &'a dyn AssociatedType<Item = &'a u32>) {
    takes_static_associated_type(arg); //~ ERROR borrowed data escapes outside of function
}

trait RepeatedLifetime<'a, 'b> {}

fn takes_repeated_lifetime<'c>(_: &(dyn RepeatedLifetime<'c, 'c> + 'static)) {}

fn mismatched_repeated_lifetime<'a, 'b>(arg: &'a dyn RepeatedLifetime<'a, 'b>) {
    //~^ ERROR one or more lifetime errors were found in this item
    takes_repeated_lifetime(arg); //~ ERROR borrowed data escapes outside of function
    //~^ ERROR borrowed data escapes outside of function
}

fn main() {}
