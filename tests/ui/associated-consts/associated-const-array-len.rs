trait Foo {
    const ID: usize;
}

const X: [i32; <i32 as Foo>::ID] = [0, 1, 2];
//~^ ERROR trait `Foo` is not implemented for `i32`

fn main() {
    assert_eq!(1, X);
}
