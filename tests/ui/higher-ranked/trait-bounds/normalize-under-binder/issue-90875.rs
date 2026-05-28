//@ check-pass

trait Variable<'a> {
    type Type;
}

impl Variable<'_> for () {
    type Type = ();
}

fn check<F, T>(_: F)
where
    F: Fn(T), // <- if removed, all fn_* then require type annotations
    F: for<'a> Fn(<T as Variable<'a>>::Type),
    T: for<'a> Variable<'a>,
{
}

fn test(arg: impl Fn(())) {
    fn fn_1(_: ()) {}
    let fn_2 = |_: ()| ();
    let fn_3 = |a| fn_1(a);
    let fn_4 = arg;

    check(fn_1); // Error
    check(fn_2); // Ok
    check(fn_3); // Ok
    check(fn_4); // Error
}

fn main() {}
