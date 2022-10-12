// Regression test for #98693.
//
// The closure encounters an obligation that `T` must outlive `!U1`,
// a placeholder from universe U1. We were ignoring this placeholder
// when promoting the constraint to the enclosing function, and
// thus incorrectly judging the closure to be safe.

fn assert_static<T>()
where
    for<'a> T: 'a,
{
}

fn test<T>() {
    || {
        //~^ ERROR the parameter type `T` may not live long enough
        assert_static::<T>();
    };
}

fn main() {}
