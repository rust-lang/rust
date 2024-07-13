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
//~^ HELP add a `use<...>` bound
    x
//~^ ERROR hidden type for
}

fn no_params_yet(_: impl Sized, y: &()) -> impl Sized {
//~^ HELP add a `use<...>` bound
    y
//~^ ERROR hidden type for
}

fn yes_params_yet<'a, T>(_: impl Sized, y: &'a ()) -> impl Sized {
//~^ HELP add a `use<...>` bound
    y
//~^ ERROR hidden type for
}

fn main() {}
