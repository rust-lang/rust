enum Foo {
    A(usize),
    B
}

fn bar() -> i32 {
    0i32
}

fn main() {
    let x = Foo::A;
    let _y = x as i32;
    let _y1 = Foo::A as i32;
    let _y = x as u32;
    let _z = bar as u32;
    let _y = bar as i64;
    let _y = bar as u64;
    let _z = Foo::A as i128;
    let _z = Foo::A as u128;
    let _z = bar as i128;
    let _z = bar as u128;
}
