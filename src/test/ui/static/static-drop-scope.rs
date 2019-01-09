#![feature(const_fn)]

struct WithDtor;

impl Drop for WithDtor {
    fn drop(&mut self) {}
}

static PROMOTION_FAIL_S: Option<&'static WithDtor> = Some(&WithDtor);
//~^ ERROR destructors cannot be evaluated at compile-time
//~| ERROR borrowed value does not live long enoug

const PROMOTION_FAIL_C: Option<&'static WithDtor> = Some(&WithDtor);
//~^ ERROR destructors cannot be evaluated at compile-time
//~| ERROR borrowed value does not live long enoug

static EARLY_DROP_S: i32 = (WithDtor, 0).1;
//~^ ERROR destructors cannot be evaluated at compile-time

const EARLY_DROP_C: i32 = (WithDtor, 0).1;
//~^ ERROR destructors cannot be evaluated at compile-time

const fn const_drop<T>(_: T) {}
//~^ ERROR destructors cannot be evaluated at compile-time

const fn const_drop2<T>(x: T) {
    (x, ()).1
    //~^ ERROR destructors cannot be evaluated at compile-time
}

fn main () {}
