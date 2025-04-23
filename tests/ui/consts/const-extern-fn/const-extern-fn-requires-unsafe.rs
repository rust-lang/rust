const unsafe extern "C" fn foo() -> usize {
    5
}

const unsafe extern "C-unwind" fn bar() -> usize {
    5
}

fn main() {
    let a: [u8; foo()];
    //~^ ERROR call to unsafe function `foo` is unsafe and requires unsafe function or block
    foo();
    //~^ ERROR call to unsafe function `foo` is unsafe and requires unsafe function or block
    let b: [u8; bar()];
    //~^ ERROR call to unsafe function `bar` is unsafe and requires unsafe function or block
    bar();
    //~^ ERROR call to unsafe function `bar` is unsafe and requires unsafe function or block
}
