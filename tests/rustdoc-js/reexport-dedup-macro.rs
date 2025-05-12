//@ aux-crate: macro_in_module=macro-in-module.rs
#![crate_name = "foo"]
extern crate macro_in_module;

// Test case based on the relationship between alloc and std.
#[doc(inline)]
pub use macro_in_module::vec;

#[macro_use]
mod hidden_macro_module {
    #[macro_export]
    macro_rules! myspecialvec {
        () => {};
    }
}
