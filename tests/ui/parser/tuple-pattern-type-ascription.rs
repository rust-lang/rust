// Writing the type of each element inline in a tuple pattern, e.g.
// `(a: bool, b: u8)`, is not valid syntax. In a `let` binding, where the pattern
// can be followed by a type, check that we recover and suggest writing the
// element types together as a tuple type after the pattern.
//
// Regression test for <https://github.com/rust-lang/rust/issues/149246>.

fn main() {
    let (a: bool,) = (true,);
    //~^ ERROR the elements of a tuple pattern cannot be given types individually

    let (b: bool, c: u8) = (true, 1);
    //~^ ERROR the elements of a tuple pattern cannot be given types individually

    let (d: bool, e) = (true, 1);
    //~^ ERROR the elements of a tuple pattern cannot be given types individually

    let (f: Option<u8>,) = (Some(1),);
    //~^ ERROR the elements of a tuple pattern cannot be given types individually

    let _ = (a, b, c, d, e, f);

    // With no whitespace after the colon we keep the existing "maybe write a
    // path separator here" suggestion instead, since `m::C` is the likelier
    // intent.
    let (m:C,) = (0,);
    //~^ ERROR expected one of
}
