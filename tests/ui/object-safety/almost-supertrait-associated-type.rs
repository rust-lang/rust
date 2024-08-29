// Test for fixed unsoundness in #126079.
// Enforces that the associated types that are object safe

use std::marker::PhantomData;

fn transmute<T, U>(t: T) -> U {
    (&PhantomData::<T> as &dyn Foo<T, U>).transmute(t)
    //~^ ERROR the trait `Foo` cannot be made into an object
    //~| ERROR the trait `Foo` cannot be made into an object
}

struct ActuallySuper;
struct NotActuallySuper;
trait Super<Q> {
    type Assoc;
}

trait Dyn {
    type Out;
}
impl<T, U> Dyn for dyn Foo<T, U> + '_ {
//~^ ERROR the trait `Foo` cannot be made into an object
    type Out = U;
}
impl<S: Dyn<Out = U> + ?Sized, U> Super<NotActuallySuper> for S {
    type Assoc = U;
}

trait Foo<T, U>: Super<ActuallySuper, Assoc = T>
where
    <Self as Mirror>::Assoc: Super<NotActuallySuper>
{
    fn transmute(&self, t: T) -> <Self as Super<NotActuallySuper>>::Assoc;
}

trait Mirror {
    type Assoc: ?Sized;
}
impl<T: ?Sized> Mirror for T {
    type Assoc = T;
}

impl<T, U> Foo<T, U> for PhantomData<T> {
    fn transmute(&self, t: T) -> T {
        t
    }
}
impl<T> Super<ActuallySuper> for PhantomData<T> {
    type Assoc = T;
}
impl<T> Super<NotActuallySuper> for PhantomData<T> {
    type Assoc = T;
}

fn main() {
    let x = String::from("hello, world");
    let s = transmute::<&str, &'static str>(x.as_str());
    drop(x);
    println!("> {s}");
}
