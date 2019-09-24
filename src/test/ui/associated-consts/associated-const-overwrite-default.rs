// run-pass

trait Foo {
    const ID: i32 = 2;
}

impl Foo for i32 {
    const ID: i32 = 1;
}

fn main() {
    assert_eq!(1, <i32 as Foo>::ID);
}
