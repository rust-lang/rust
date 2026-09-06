//@ no-prefer-dynamic

#![feature(lang_items, panic_unwind, rustc_attrs)]
#![crate_type = "rlib"]
#![no_std]

extern crate unwind;

pub struct DerefsToF64(f64);

impl core::ops::Deref for DerefsToF64 {
    type Target = f64;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

mod inner {
    impl f64 {
        /// [f64::clone]
        #[rustc_allow_incoherent_impl]
        pub fn method() {}
    }
}

#[panic_handler]
fn bar(_: &core::panic::PanicInfo) -> ! { loop {} }
