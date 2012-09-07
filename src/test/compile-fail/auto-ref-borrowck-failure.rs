// Tests that auto-ref can't create mutable aliases to immutable memory.

struct Foo {
    x: int
}

trait Stuff {
    fn printme();
}

impl &mut Foo : Stuff {
    fn printme() {
        io::println(fmt!("%d", self.x));
    }
}

fn main() {
    let x = Foo { x: 3 };
    x.printme();    //~ ERROR illegal borrow
}

