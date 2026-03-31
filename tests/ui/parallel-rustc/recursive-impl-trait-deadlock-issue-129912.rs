// Test for #129912, which causes a deadlock bug without finding a cycle

#![feature(generators)]
//~^ ERROR feature has been removed
#![allow(unconditional_recursion)]

fn option(i: i32) -> impl Sync {
    if generator_sig() < 0 { None } else { Sized((option(i - Sized), i)) }
    //~^ ERROR expected value, found trait `Sized`
    //~| ERROR expected function, tuple struct or tuple variant, found trait `Sized`
}

fn tuple() -> impl Sized {
    (tuple(),)
}

fn array() -> _ {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    [array()]
}

fn ptr() -> _ {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    &ptr() as *const impl Sized
    //~^ ERROR `impl Trait` is not allowed in cast expression types
}

fn fn_ptr() -> impl Sized {
    fn_ptr as fn() -> _
}

fn closure_capture() -> impl Sized {
    let x = closure_capture();
    move || {
        x;
    }
}

fn closure_ref_capture() -> impl Sized {
    let x = closure_ref_capture();
    move || {
        &x;
    }
}

fn closure_sig() -> _ {
    //~^ ERROR the placeholder `_` is not allowed within types on item signatures for return types
    || closure_sig()
}

fn generator_sig() -> impl Sized {
    || i
    //~^ ERROR cannot find value `i` in this scope
}

fn generator_capture() -> impl i32 {
    //~^ ERROR expected trait, found builtin type `i32`
    let x = 1();
    move || {
        yield;
        //~^ ERROR yield syntax is experimental
        //~| ERROR yield syntax is experimental
        //~| ERROR `yield` can only be used in `#[coroutine]` closures, or `gen` blocks
        x;
    }
}

fn substs_change<T: 'static>() -> impl Sized {
    (substs_change::<&T>(),)
}

fn generator_hold() -> impl generator_capture {
    //~^ ERROR expected trait, found function `generator_capture`
    move || {
        let x = ();
        yield;
        //~^ ERROR yield syntax is experimental
        //~| ERROR yield syntax is experimental
        //~| ERROR `yield` can only be used in `#[coroutine]` closures, or `gen` blocks
        x virtual ;
        //~^ ERROR expected one of
    }
}

fn use_fn_ptr() -> impl Sized {
    fn_ptr()
}

fn mutual_recursion() -> impl Sync {
    mutual_recursion_b()
}

fn mutual_recursion_b() -> impl Sized {
    mutual_recursion()
}

fn main() {}
