//@ no-prefer-dynamic
//@ compile-flags: -Cmetadata=aux
#![crate_type = "rlib"]
#![doc(html_root_url = "http://example.com/")]
#![feature(rustc_attrs)]
#![feature(lang_items)]
#![feature(panic_unwind)]
#![no_std]

extern crate unwind;

#[panic_handler]
fn bar(_: &core::panic::PanicInfo) -> ! { loop {} }

/// dox
#[rustc_doc_primitive = "pointer"]
const _: () = ();
