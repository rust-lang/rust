// The double space in `impl  Iterator` is load bearing! We want to make sure we don't regress by
// accident if the internal string representation changes.
#[rustfmt::skip]
fn foo(constraints: impl  Iterator) {
    for constraint in constraints {
        qux(constraint);
//~^ ERROR `<impl Iterator as Iterator>::Item` doesn't implement `Debug`
    }
}

fn bar<T>(t: T, constraints: impl Iterator) where T: std::fmt::Debug {
    for constraint in constraints {
        qux(t);
        qux(constraint);
//~^ ERROR `<impl Iterator as Iterator>::Item` doesn't implement `Debug`
    }
}

fn baz(t: impl std::fmt::Debug, constraints: impl Iterator) {
    for constraint in constraints {
        qux(t);
        qux(constraint);
//~^ ERROR `<impl Iterator as Iterator>::Item` doesn't implement `Debug`
    }
}

fn bat<I, T: std::fmt::Debug>(t: T, constraints: impl Iterator, _: I) {
    for constraint in constraints {
        qux(t);
        qux(constraint);
//~^ ERROR `<impl Iterator as Iterator>::Item` doesn't implement `Debug`
    }
}

fn bak(constraints: impl  Iterator + std::fmt::Debug) {
    for constraint in constraints {
        qux(constraint);
//~^ ERROR `<impl Iterator + std::fmt::Debug as Iterator>::Item` doesn't implement
    }
}

#[rustfmt::skip]
fn baw<>(constraints: impl Iterator) {
    for constraint in constraints {
        qux(constraint);
//~^ ERROR `<impl Iterator as Iterator>::Item` doesn't implement `Debug`
    }
}

fn qux(_: impl std::fmt::Debug) {}

fn main() {}
