// compile-flags: -Z continue-parse-after-error

mod Foo {
    /// document
    //~^ ERROR expected item after doc comment
}

fn main() {}
