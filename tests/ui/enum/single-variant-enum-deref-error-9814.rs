// https://github.com/rust-lang/rust/issues/9814
// Verify that single-variant enums can't be de-referenced
// Regression test for issue #9814

enum Foo { Bar(isize) }

fn main() {
    let _ = *Foo::Bar(2); //~ ERROR type `Foo` cannot be dereferenced
}
