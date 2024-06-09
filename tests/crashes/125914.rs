//@ known-bug: rust-lang/rust#125914
enum AstKind<'ast> {
    ExprInt,
}

enum Foo {
    Bar(isize),
    Baz,
}

enum Other {
    Other1(Foo),
    Other2(AstKind),
}

fn main() {
    match Other::Other1(Foo::Baz) {
        ::Other::Other2(::Foo::Bar(..)) => {}
    }
}
