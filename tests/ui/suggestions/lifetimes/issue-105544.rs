//@ run-rustfix

#![allow(warnings)]

fn foo(d: impl Sized, p: &mut ()) -> impl Sized + '_ { //~ NOTE the parameter type `impl Sized` must be valid for the anonymous lifetime defined here...
//~^ HELP consider adding an explicit lifetime bound
    (d, p)
    //~^ ERROR the parameter type `impl Sized` may not live long enough
    //~| NOTE ...so that the type `impl Sized` will meet its required lifetime bounds
}

fn foo1<'b>(d: impl Sized, p: &'b mut ()) -> impl Sized + '_ {
//~^ NOTE the parameter type `impl Sized` must be valid for the lifetime `'b` as defined here...
//~| HELP consider adding an explicit lifetime bound
    (d, p) //~ NOTE ...so that the type `impl Sized` will meet its required lifetime bounds
    //~^ ERROR the parameter type `impl Sized` may not live long enough
}

fn foo2<'a>(d: impl Sized + 'a, p: &mut ()) -> impl Sized + '_ { //~ NOTE the parameter type `impl Sized + 'a` must be valid for the anonymous lifetime defined here...
//~^ HELP consider adding an explicit lifetime bound
    (d, p)
    //~^ ERROR the parameter type `impl Sized + 'a` may not live long enough
    //~| NOTE ...so that the type `impl Sized + 'a` will meet its required lifetime bounds
}

fn bar<T : Sized>(d: T, p: & mut ()) -> impl Sized + '_ { //~ NOTE the parameter type `T` must be valid for the anonymous lifetime defined here...
//~^ HELP consider adding an explicit lifetime bound
    (d, p)
    //~^ ERROR the parameter type `T` may not live long enough
    //~| NOTE ...so that the type `T` will meet its required lifetime bounds
}

fn bar1<'b, T : Sized>(d: T, p: &'b mut ()) -> impl Sized + '_ {
//~^ NOTE the parameter type `T` must be valid for the lifetime `'b` as defined here...
//~| HELP consider adding an explicit lifetime bound
    (d, p) //~ NOTE ...so that the type `T` will meet its required lifetime bounds
    //~^ ERROR the parameter type `T` may not live long enough
}

fn bar2<'a, T : Sized + 'a>(d: T, p: &mut ()) -> impl Sized + '_ { //~ NOTE the parameter type `T` must be valid for the anonymous lifetime defined here...
//~^ HELP consider adding an explicit lifetime bound
    (d, p)
    //~^ ERROR the parameter type `T` may not live long enough
    //~| NOTE ...so that the type `T` will meet its required lifetime bounds
}

fn main() {}
