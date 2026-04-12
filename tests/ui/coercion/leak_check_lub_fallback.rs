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
    // The lub should occur inside of dead code so that we
    // can be sure we are actually testing whether we leak
    // checked.
    loop {}

    let lhs = mk::<Foo<for<'a> fn(&'static (), &'a ())>>();
    let rhs = mk::<Foo<for<'a> fn(&'a (), &'static ())>>();
    lub!(lhs, rhs);
    //~^ ERROR: `if` and `else` have incompatible types
}

fn main() {}
