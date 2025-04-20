//! This test used to ICE: rust-lang/rust#125914
//! Instead of actually analyzing the erroneous patterns,
//! we instead stop after typeck where errors are already
//! reported.

enum AstKind<'ast> {
    //~^ ERROR: `'ast` is never used
    ExprInt,
}

enum Foo {
    Bar(isize),
    Baz,
}

enum Other {
    Other1(Foo),
    Other2(AstKind), //~ ERROR: missing lifetime specifier
}

fn main() {
    match Other::Other1(Foo::Baz) {
        crate::Other::Other2(crate::Foo::Bar(..)) => {}
    }
}
