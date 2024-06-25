//@ known-bug: unknown

// RPITITs don't have variances in their GATs, so they always relate invariantly
// and act as if they capture all their args.
// To fix this soundly, we need to make sure that all the trait header args
// remain captured, since they affect trait selection.

#![feature(precise_capturing)]

trait Foo<'a> {
    fn hello() -> impl PartialEq + use<Self>;
}

fn test<'a, 'b, T: for<'r> Foo<'r>>() {
    PartialEq::eq(
        &<T as Foo<'a>>::hello(),
        &<T as Foo<'b>>::hello(),
    );
}

fn main() {}
