// Regression test for #132767
// This used to ICE with "maybe try to call `try_normalize_erasing_regions`"
// when using associated types with fn pointers and impl Trait.
// Now it correctly reports a trait bound error.

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
