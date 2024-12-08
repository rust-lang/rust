//@ run-pass
trait Foo {
    type T;
}

impl Foo for i32 {
    type T = isize;
}

fn main() {
    let x: <i32 as Foo>::T = 22;
    let y: isize = 44;
    assert_eq!(x * 2, y);
}
