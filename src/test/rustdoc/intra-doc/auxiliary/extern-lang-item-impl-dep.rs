// no-prefer-dynamic

#![feature(lang_items)]

#![crate_type = "rlib"]
#![no_std]

pub struct DerefsToF64(f64);

impl core::ops::Deref for DerefsToF64 {
    type Target = f64;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

mod inner {
    #[lang = "f64_runtime"]
    impl f64 {
        /// [f64::clone]
        pub fn method() {}
    }
}

#[lang = "eh_personality"]
fn foo() {}

#[panic_handler]
fn bar(_: &core::panic::PanicInfo) -> ! { loop {} }
