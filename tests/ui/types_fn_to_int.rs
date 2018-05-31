enum Foo {
    A(usize),
    B
}

fn bar() -> i32 {
    0i32
}

fn main() {
    let x = Foo::A;
    let y = x as i32;
    let y1 = Foo::A as i32;

    let z = bar as u32;
}
