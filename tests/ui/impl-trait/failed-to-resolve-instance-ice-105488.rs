// ICE failed to resolve instance for <fn() -> impl MyFnOnce  ...
// issue: rust-lang/rust#105488
//@ build-fail
//~^^^ ERROR overflow evaluating the requirement `fn() -> impl MyFnOnce

pub trait MyFnOnce {
    type Output;

    fn call_my_fn_once(self) -> Self::Output;
}

pub struct WrapFnOnce<F>(F);

impl<F: FnOnce() -> D, D: MyFnOnce> MyFnOnce for WrapFnOnce<F> {
    type Output = D::Output;

    fn call_my_fn_once(self) -> Self::Output {
        D::call_my_fn_once(self.0())
    }
}

impl<F: FnOnce() -> D, D: MyFnOnce> MyFnOnce for F {
    type Output = D::Output;

    fn call_my_fn_once(self) -> Self::Output {
        D::call_my_fn_once(self())
    }
}

pub fn my_fn_1() -> impl MyFnOnce {
    my_fn_2
}

pub fn my_fn_2() -> impl MyFnOnce {
    WrapFnOnce(my_fn_1)
}

fn main() {
    let v = my_fn_1();

    let _ = v.call_my_fn_once();
}
