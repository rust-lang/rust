// compile-flags: -Z parse-only -Z continue-parse-after-error

mod Foo {
    /// document
    //~^ ERROR expected item after doc comment
}
