//@revisions: edition2015 edition2024
//@[edition2015] edition:2015
//@[edition2024] edition:2024
fn lifetime<'a, 'b>(x: &'a ()) -> impl Sized + use<'b> {
//~^ HELP add `'a` to the `use<...>` bound
    x
//~^ ERROR hidden type for
}

fn param<'a, T>(x: &'a ()) -> impl Sized + use<T> {
//~^ HELP add `'a` to the `use<...>` bound
    x
//~^ ERROR hidden type for
}

fn empty<'a>(x: &'a ()) -> impl Sized + use<> {
//~^ HELP add `'a` to the `use<...>` bound
    x
//~^ ERROR hidden type for
}

trait Captures<'a> {}
impl<T> Captures<'_> for T {}

fn missing<'a, 'captured, 'not_captured, Captured>(x: &'a ()) -> impl Captures<'captured> {
//[edition2015]~^ HELP add a `use<...>` bound
    x
//[edition2015]~^ ERROR hidden type for
}

fn no_params_yet(_: impl Sized, y: &()) -> impl Sized {
//[edition2015]~^ HELP add a `use<...>` bound
    y
//[edition2015]~^ ERROR hidden type for
}

fn yes_params_yet<'a, T>(_: impl Sized, y: &'a ()) -> impl Sized {
//[edition2015]~^ HELP add a `use<...>` bound
    y
//[edition2015]~^ ERROR hidden type for
}

fn main() {}
