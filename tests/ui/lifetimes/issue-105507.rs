//@ run-rustfix
//
#![allow(warnings)]
struct Wrapper<'a, T: ?Sized>(&'a T);

trait Project {
    type Projected<'a> where Self: 'a;
    fn project(this: Wrapper<'_, Self>) -> Self::Projected<'_>;
}
trait MyTrait {}
trait ProjectedMyTrait {}

impl<T> Project for Option<T> {
    type Projected<'a> = Option<Wrapper<'a, T>> where T: 'a;
    fn project(this: Wrapper<'_, Self>) -> Self::Projected<'_> {
        this.0.as_ref().map(Wrapper)
    }
}

impl<T: MyTrait> MyTrait for Option<Wrapper<'_, T>> {}

impl<T: ProjectedMyTrait> MyTrait for Wrapper<'_, T> {}

impl<T> ProjectedMyTrait for T
    where
        T: Project,
        for<'a> T::Projected<'a>: MyTrait,
        //~^ NOTE due to current limitations in the borrow checker, this implies a `'static` lifetime
        //~| NOTE due to current limitations in the borrow checker, this implies a `'static` lifetime
{}

fn require_trait<T: MyTrait>(_: T) {}

fn foo<T : MyTrait, U : MyTrait>(wrap: Wrapper<'_, Option<T>>, wrap1: Wrapper<'_, Option<U>>) {
    //~^ HELP consider restricting the type parameter to the `'static` lifetime
    //~| HELP consider restricting the type parameter to the `'static` lifetime
    require_trait(wrap);
    //~^ ERROR `T` does not live long enough
    require_trait(wrap1);
    //~^ ERROR `U` does not live long enough
}

fn main() {}
