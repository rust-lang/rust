trait Foo {
    const ID: i32;
}

const X: i32 = <i32>::ID;
//~^ ERROR no associated item named `ID` found for type `i32`

fn main() {
    assert_eq!(1, X);
}
