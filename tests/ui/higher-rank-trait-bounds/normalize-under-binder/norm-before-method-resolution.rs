// check-fail
// known-bug: #89196

// Should pass, but we normalize and check bounds before we resolve the generics
// of the function (which we know because of the return type).

trait Trait<'a> {
    type Out;
}

impl<'a, T> Trait<'a> for T {
    type Out = T;
}

fn weird_bound<X>() -> X
    where
        for<'a> X: Trait<'a>,
        for<'a> <X as Trait<'a>>::Out: Copy
{ todo!() }

fn main() {
    let _: () = weird_bound();
}
