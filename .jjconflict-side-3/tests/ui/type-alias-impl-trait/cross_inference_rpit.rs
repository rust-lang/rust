//@ check-pass

fn foo(b: bool) -> impl Copy {
    if b {
        return (5,6)
    }
    let x: (_, _) = foo(true);
    println!("{:?}", x);
    (1u32, 2u32)
}

fn main() {
    foo(false);
}
