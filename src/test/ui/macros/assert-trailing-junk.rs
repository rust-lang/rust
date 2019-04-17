// Ensure assert macro does not ignore trailing garbage.
//
// See https://github.com/rust-lang/rust/issues/60024 for details.

fn main() {
    assert!(true some extra junk, "whatever");
    //~^ ERROR expected one of

    assert!(true some extra junk);
    //~^ ERROR expected one of

    assert!(true, "whatever" blah);
    //~^ ERROR no rules expected
}
