// Regression test for #132767. Coercing the fn item `bar` to a fn pointer while
// `Foo::value` holds the associated-type projection `<<T as Func>::Ret as Id>::Assoc`
// used to ICE during normalization ("maybe try to call `try_normalize_erasing_regions`
// instead"). It should now report the ordinary `u32: Id` trait error instead of crashing.

use std::hint::black_box;

trait Func {
    type Ret: Id;
}

trait Id {
    type Assoc;
}

impl Id for i32 {
    type Assoc = i32;
}

impl<F: FnOnce() -> R, R: Id> Func for F {
    type Ret = R;
}

fn bar() -> impl Copy + Id {
    //~^ ERROR the trait bound `u32: Id` is not satisfied
    0u32
}

struct Foo<T: Func> {
    _func: T,
    value: Option<<<T as Func>::Ret as Id>::Assoc>,
}

fn main() {
    let mut fn_def = black_box(Foo {
        _func: bar,
        value: None,
    });
    let fn_ptr = black_box(Foo {
        _func: bar as fn() -> _,
        value: None,
    });
    fn_def.value = fn_ptr.value;
    black_box(fn_def);
}
