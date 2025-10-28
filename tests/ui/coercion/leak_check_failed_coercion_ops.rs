macro_rules! lub {
    ($lhs:expr, $rhs:expr) => {
        if true { $lhs } else { $rhs }
    };
}

struct Foo<T>(T);

fn mk<T>() -> T {
    loop {}
}

fn lub_deep_binder() {
    loop {}

    let a: Foo<for<'a> fn(&'a ())> = mk::<Foo<fn(&'static ())>>();
    //~^ ERROR: mismatched types

    let lhs = mk::<Foo<for<'a> fn(&'static (), &'a ())>>();
    let rhs = mk::<Foo<for<'a> fn(&'a (), &'static ())>>();
    lub!(lhs, rhs);
    //~^ ERROR: `if` and `else` have incompatible types
}

fn main() {}
