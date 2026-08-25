//@ check-pass
//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver

pub trait Fn0: Fn(i32) -> Self::Out {
    type Out;
}

impl<F: Fn(i32) -> ()> Fn0 for F {
    type Out = ();
}

pub fn closure_typer(_: impl Fn0) {}

fn main() {
    closure_typer(move |x| {
        let _: i64 = x.into();
    });
}
