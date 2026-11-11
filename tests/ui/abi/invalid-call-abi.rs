// Tests the `"rustc-invalid"` ABI, which is never canonizable.

#![feature(rustc_attrs)]

const extern "rust-invalid" fn foo() {
    //~^ ERROR "rust-invalid" is not a supported ABI for the current target
    panic!()
}

fn main() {
    foo();
}
