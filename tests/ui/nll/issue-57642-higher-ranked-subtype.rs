// Regression test for issue #57642
// Tests that we reject a bad higher-ranked subtype

trait X {
    type G;
    fn make_g() -> Self::G;
}

impl<'a> X for fn(&'a ()) {
    type G = &'a ();

    fn make_g() -> Self::G {
        &()
    }
}

trait Y {
    type F;
    fn make_f() -> Self::F;
}

impl<T> Y for fn(T) {
    type F = fn(T);

    fn make_f() -> Self::F {
        |_| {}
    }
}

fn higher_ranked_region_has_lost_its_binder() {
    let x = <fn (&())>::make_g();
    //~^ ERROR no function or associated item named `make_g` found for fn pointer `for<'a> fn(&'a ())` in the current scope
}

fn magical() {
    let x = <fn (&())>::make_f(); //~ ERROR no function
}

fn main() {}
