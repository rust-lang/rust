trait Foo {
    const ID: i32;
}

const X: i32 = <i32>::ID;
//~^ ERROR no associated item named `ID` found

fn main() {
    assert_eq!(1, X);
}
