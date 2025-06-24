// Test for https://github.com/rust-lang/rust/issues/86232
// Due to AST-to-HIR lowering nuances, we used to allow unsupported ABIs to "leak" into the HIR
// without being checked, as we would check after generating the ExternAbi.
//
// Here we test that an unsupported ABI in various impl-related positions will be rejected,
// both in the original declarations and the actual implementations.

#![feature(rustc_attrs)]
//@ compile-flags: --crate-type lib

pub struct FnPtrBearer {
    pub ptr: extern "rust-invalid" fn(),
    //~^ ERROR: is not a supported ABI
}

impl FnPtrBearer {
    pub extern "rust-invalid" fn inherent_fn(self) {
        //~^ ERROR: is not a supported ABI
        (self.ptr)()
    }
}

pub trait Trait {
    extern "rust-invalid" fn trait_fn(self);
    //~^ ERROR: is not a supported ABI
}

impl Trait for FnPtrBearer {
    extern "rust-invalid" fn trait_fn(self) {
        //~^ ERROR: is not a supported ABI
        self.inherent_fn()
    }
}
