// Fix for #142969 where an invalid ABI in a signature still had its call ABI computed
// because CTFE tried to evaluate it, despite previous errors during AST-to-HIR lowering.

#![feature(rustc_attrs)]

const extern "rust-invalid" fn foo() {
    //~^ ERROR "rust-invalid" is not a supported ABI for the current target
    panic!()
}

const _: () = foo();


fn main() {}
