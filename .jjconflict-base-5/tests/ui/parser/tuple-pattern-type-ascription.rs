// Writing the type of each element inline in a tuple pattern, e.g.
// `(a: bool, b: u8)`, is not valid syntax. Check that we recover from it, and
// that in a `let` binding, where the pattern can be followed by a type, we also
// suggest writing the element types together as a tuple type after the pattern.
//
// Regression test for <https://github.com/rust-lang/rust/issues/149246>.

struct S {
    f: (bool, u8),
}

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

    // A parenthesized pattern is not a one-element tuple, so its type isn't a
    // tuple type either.
    let (g: bool) = true;
    //~^ ERROR a parenthesized pattern cannot be given a type

    // The pattern is already followed by a type, so we only suggest removing the
    // inline annotations.
    let (h: bool, i): (bool, u8) = (true, 1);
    //~^ ERROR the elements of a tuple pattern cannot be given types individually

    // A rest pattern matches any number of elements, so we don't know how many
    // types the tuple type would have to list.
    let (.., j: u8) = (true, 1);
    //~^ ERROR the elements of a tuple pattern cannot be given types individually

    // Anywhere other than the top level pattern of a `let` binding we recover
    // but don't suggest anything.
    match (true, 1) {
        (k: bool, l: u8) => {}
        //~^ ERROR the elements of a tuple pattern cannot be given types individually
    }

    for (n: bool, o: u8) in [(true, 1)] {}
    //~^ ERROR the elements of a tuple pattern cannot be given types individually

    if let (p: bool, q: u8) = (true, 1) {}
    //~^ ERROR the elements of a tuple pattern cannot be given types individually

    let [(r: bool,)] = [(true,)];
    //~^ ERROR the elements of a tuple pattern cannot be given types individually

    let S { f: (s: bool, t: u8) } = S { f: (true, 1) };
    //~^ ERROR the elements of a tuple pattern cannot be given types individually

    // With no whitespace after the colon we keep the existing "maybe write a
    // path separator here" suggestion instead, since `m::C` is the likelier
    // intent.
    let (m:C,) = (0,);
    //~^ ERROR expected one of
}

fn param((u: bool, v: u8): (bool, u8)) {}
//~^ ERROR the elements of a tuple pattern cannot be given types individually
