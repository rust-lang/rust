// no-prefer-dynamic
// compile-flags: -Cmetadata=aux
#![crate_type = "rlib"]
#![doc(html_root_url = "http://example.com/")]
#![feature(lang_items)]
#![no_std]

#[lang = "eh_personality"]
fn foo() {}

#[panic_handler]
fn bar(_: &core::panic::PanicInfo) -> ! { loop {} }

/// dox
#[doc(primitive = "pointer")]
pub mod ptr {}
