// Test for issue #57362, ensuring that the self ty is shown in cases of higher-ranked lifetimes
// conflicts: the `expected` and `found` trait refs would otherwise be printed the same, leading
// to confusing notes such as:
//  = note: expected type `Trait`
//             found type `Trait`

// extracted from a similar issue: #57642
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

// FIXME(@compiler-errors): This error message is less than helpful.
fn g() {
    let x = <fn (&())>::make_g();
    //~^ ERROR no function or associated item named `make_g` found for fn pointer `for<'a> fn(&'a ())` in the current scope
}

fn main() {}
