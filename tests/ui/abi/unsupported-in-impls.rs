// Test for https://github.com/rust-lang/rust/issues/86232
// Due to AST-to-HIR lowering nuances, we used to allow unsupported ABIs to "leak" into the HIR
// without being checked, as we would check after generating the ExternAbi.
// Checking afterwards only works if we examine every HIR construct that contains an ExternAbi,
// and those may be very different in HIR, even if they read the same in source.
// This made it very easy to make mistakes.
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
