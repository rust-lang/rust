// https://github.com/rust-lang/rust/issues/82446
// Spurious 'help: store this in the heap' regression test
trait MyTrait {}

struct Foo {
    val: Box<dyn MyTrait>
}

fn make_it(val: &Box<dyn MyTrait>) {
    Foo {
        val //~ ERROR [E0308]
    };
}

fn main() {}
