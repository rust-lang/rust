#![feature(const_extern_fn)]

const unsafe extern "C" fn foo() -> usize { 5 }

fn main() {
    let a: [u8; foo()];
    //~^ ERROR call to unsafe function is unsafe and requires unsafe function or block
    foo();
    //~^ ERROR call to unsafe function is unsafe and requires unsafe function or block
}
