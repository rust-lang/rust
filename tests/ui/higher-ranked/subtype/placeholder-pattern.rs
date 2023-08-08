// check-pass
// Check that higher ranked subtyping correctly works when using
// placeholder patterns.
fn hr_subtype<'c>(f: for<'a, 'b> fn(&'a (), &'b ())) {
    let _: for<'a> fn(&'a (), &'a ()) = f;
    let _: for<'a, 'b> fn(&'a (), &'b ()) = f;
    let _: for<'a> fn(&'a (), &'c ()) = f;
    let _: fn(&'c (), &'c ()) = f;
}

fn simple<'c>(x: (&'static i32,)) {
    let _: (&'c i32,) = x;
}

fn main() {
    hr_subtype(|_, _| {});
    simple((&3,));
}
