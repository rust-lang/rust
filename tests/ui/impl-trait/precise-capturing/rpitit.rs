// RPITITs don't have variances in their GATs, so they always relate invariantly
// and act as if they capture all their args.
// To fix this soundly, we need to make sure that all the trait header args
// remain captured, since they affect trait selection.

fn eq_types<T>(_: T, _: T) {}

trait TraitLt<'a: 'a> {
    fn hello() -> impl Sized + use<Self>;
    //~^ ERROR `impl Trait` captures lifetime parameter, but it is not mentioned in `use<...>` precise captures list
}
fn trait_lt<'a, 'b, T: for<'r> TraitLt<'r>> () {
    eq_types(
        //~^ ERROR lifetime may not live long enough
        //~| ERROR lifetime may not live long enough
        <T as TraitLt<'a>>::hello(),
        <T as TraitLt<'b>>::hello(),
    );
}

trait MethodLt {
    fn hello<'a: 'a>() -> impl Sized + use<Self>;
}
fn method_lt<'a, 'b, T: MethodLt> () {
    eq_types(
        T::hello::<'a>(),
        T::hello::<'b>(),
    );
    // Good!
}

fn main() {}
