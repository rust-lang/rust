//@ revisions: with-generic-asset without-generic-asset
//@ [with-generic-asset] compile-flags: --cfg feature="generic_assert"

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

    assert!(true "whatever" blah);
    //~^ ERROR unexpected string literal
    //~^^ ERROR no rules expected

    assert!(true;);
    //~^ ERROR macro requires an expression

    assert!(false || true "error message");
    //~^ ERROR unexpected string literal
}
