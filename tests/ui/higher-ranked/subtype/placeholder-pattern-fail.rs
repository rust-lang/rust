// Check that incorrect higher ranked subtyping
// causes an error.
struct Inv<'a>(fn(&'a ()) -> &'a ());
fn hr_subtype<'c>(f: for<'a, 'b> fn(Inv<'a>, Inv<'a>)) {
    // ok
    let _: for<'a> fn(Inv<'a>, Inv<'a>) = f;
    let sub: for<'a> fn(Inv<'a>, Inv<'a>) = f;
    // no
    let _: for<'a, 'b> fn(Inv<'a>, Inv<'b>) = sub;
    //~^ ERROR mismatched types
}

fn simple1<'c>(x: (&'c i32,)) {
    let _x: (&'static i32,) = x;
    //~^ ERROR: lifetime may not live long enough
}

fn simple2<'c>(x: (&'c i32,)) {
    let _: (&'static i32,) = x;
    //~^ ERROR: lifetime may not live long enough
}

fn main() {
    hr_subtype(|_, _| {});
    simple1((&3,));
    simple2((&3,));
}
