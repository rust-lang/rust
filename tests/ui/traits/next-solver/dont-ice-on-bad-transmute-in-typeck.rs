//@ compile-flags: -Znext-solver

trait Trait<'a> {
    type Assoc;
}

fn foo(x: for<'a> fn(<() as Trait<'a>>::Assoc)) {
    //~^ ERROR the trait bound `for<'a> (): Trait<'a>` is not satisfied
    //~| ERROR the trait bound `for<'a> (): Trait<'a>` is not satisfied
    //~| ERROR the trait bound `for<'a> (): Trait<'a>` is not satisfied
    unsafe { std::mem::transmute::<_, ()>(x); }
    //~^ ERROR the trait bound `for<'a> (): Trait<'a>` is not satisfied
    //~| ERROR the trait bound `for<'a> (): Trait<'a>` is not satisfied
    //~| ERROR the trait bound `for<'a> (): Trait<'a>` is not satisfied
}

fn main() {}
