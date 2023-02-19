// Issue #53114: NLL's borrow check had some deviations from the old borrow
// checker, and both had some deviations from our ideal state. This test
// captures the behavior of how `_` bindings are handled with respect to how we
// flag expressions that are meant to request unsafe blocks.

#[derive(Copy, Clone)]
struct I(i64);
#[derive(Copy, Clone)]
struct F(f64);

union U { a: I, b: F }

#[repr(packed)]
struct P {
    a: &'static i8,
    b: &'static u32,
}

fn let_wild_gets_unsafe_field() {
    let u1 = U { a: I(0) };
    let u2 = U { a: I(1) };
    let p = P { a: &2, b: &3 };
    let _ = &p.b;  //~ ERROR    reference to packed field
    let _ = u1.a;  // #53114: should eventually signal error as well
    let _ = &u2.a; //~ ERROR  [E0133]

    // variation on above with `_` in substructure
    let (_,) = (&p.b,);  //~ ERROR     reference to packed field
    let (_,) = (u1.a,);  //~ ERROR   [E0133]
    let (_,) = (&u2.a,); //~ ERROR   [E0133]
}

fn match_unsafe_field_to_wild() {
    let u1 = U { a: I(0) };
    let u2 = U { a: I(1) };
    let p = P { a: &2, b: &3 };
    match &p.b  { _ => { } } //~ ERROR     reference to packed field
    match u1.a  { _ => { } } //~ ERROR   [E0133]
    match &u2.a { _ => { } } //~ ERROR   [E0133]

    // variation on above with `_` in substructure
    match (&p.b,)  { (_,) => { } } //~ ERROR     reference to packed field
    match (u1.a,)  { (_,) => { } } //~ ERROR   [E0133]
    match (&u2.a,) { (_,) => { } } //~ ERROR   [E0133]
}

fn main() { }
