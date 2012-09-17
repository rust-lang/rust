enum Foo = uint;

fn main() {
    let x = Foo(1);
    let y = fmt!("%?", x);
    assert y == ~"Foo(1)";
}
