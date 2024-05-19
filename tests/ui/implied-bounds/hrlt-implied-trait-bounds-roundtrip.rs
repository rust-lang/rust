//@ check-pass
struct Foo<'a>(&'a ())
where
    (): Trait<'a>;

trait Trait<'a> {
    fn id<T>(value: &'a T) -> &'static T;
}

impl Trait<'static> for () {
    fn id<T>(value: &'static T) -> &'static T {
        value
    }
}

fn could_use_implied_bounds<'a, T>(_: Foo<'a>, x: &'a T) -> &'static T
where
    (): Trait<'a>, // This could be an implied bound
{
    <()>::id(x)
}

fn main() {
    let bar: for<'a, 'b> fn(Foo<'a>, &'b ()) = |_, _| {};

    // If `could_use_implied_bounds` were to use implied bounds,
    // keeping 'a late-bound, then we could assign that function
    // to this variable.
    let bar: for<'a> fn(Foo<'a>, &'a ()) = bar;

    // In this case, the subtyping relation here would be unsound,
    // allowing us to transmute lifetimes. This currently compiles
    // because we incorrectly deal with implied bounds inside of binders.
    let _bar: for<'a, 'b> fn(Foo<'a>, &'b ()) = bar;
}
