#![feature(const_fn)]

struct WithDtor;

impl Drop for WithDtor {
    fn drop(&mut self) {}
}

static PROMOTION_FAIL_S: Option<&'static WithDtor> = Some(&WithDtor);
//~^ ERROR destructors cannot be evaluated at compile-time
//~| ERROR temporary value dropped while borrowed

const PROMOTION_FAIL_C: Option<&'static WithDtor> = Some(&WithDtor);
//~^ ERROR destructors cannot be evaluated at compile-time
//~| ERROR temporary value dropped while borrowed

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

const EARLY_DROP_C_OPTION: i32 = (Some(WithDtor), 0).1;
//~^ ERROR destructors cannot be evaluated at compile-time

const HELPER: Option<WithDtor> = Some(WithDtor);

const EARLY_DROP_C_OPTION_CONSTANT: i32 = (HELPER, 0).1;
//~^ ERROR destructors cannot be evaluated at compile-time

fn main () {}
