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

    let z = bar as u32;

    //let c = || {0i32};
    //let ac = c as u32;
}
