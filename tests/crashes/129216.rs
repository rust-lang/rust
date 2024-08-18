//@ known-bug: rust-lang/rust#129216

trait Mirror {
    type Assoc;
}

struct Foo;

fn main() {
    <Foo as Mirror>::Assoc::new();
}
