struct Foo {
    a: u8,
    b: u8,
}

fn main() {
    let x = Foo { a:1, b:2 };
    let Foo { a: x, a: y, b: 0 } = x;
    //~^ ERROR field `a` bound multiple times in the pattern
}
