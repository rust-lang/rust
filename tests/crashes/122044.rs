//@ known-bug: #122044
use std::hint::black_box;

trait Func {
    type Ret: Id;
}

trait Id {
    type Assoc;
}
impl Id for u32 {}
impl Id for u32 {}

impl<F: FnOnce() -> R, R: Id> Func for F {
    type Ret = R;
}

fn bar() -> impl Copy + Id {
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
}
