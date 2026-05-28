// Regression test for #105009. the issue here was that even after the `PostAnalysisNormalize` pass,
// `validate` still used `Reveal::UserFacing`. This meant that it now ends up comparing
// opaque types with their revealed version, resulting in an ICE.
//
// We're using these flags to run the `PostAnalysisNormalize` pass while making it less likely to
// accidentally removing the assignment from `Foo<fn_ptr>` to `Foo<fn_def>`.

//@ compile-flags: -Zinline_mir=yes -Zmir-opt-level=0 -Zvalidate-mir
//@ run-pass

use std::hint::black_box;

trait Func {
    type Ret: Id;
}

trait Id {
    type Assoc;
}
impl Id for u32 {
    type Assoc = u32;
}
impl Id for i32 {
    type Assoc = i32;
}

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
    black_box(fn_def);
}
