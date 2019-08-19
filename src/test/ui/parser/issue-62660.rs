// Regression test for issue #62660: if a receiver's type does not
// successfully parse, emit the correct error instead of ICE-ing the compiler.

struct Foo;

impl Foo {
    pub fn foo(_: i32, self: Box<Self) {}
    //~^ ERROR expected one of `!`, `(`, `+`, `,`, `::`, `<`, or `>`, found `)`
}

fn main() {}
