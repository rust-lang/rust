// Call `[NSObject class]` using `objc::class!` and `objc::selector!`.

//@ edition: 2024
//@ only-apple
//@ run-pass

#![feature(darwin_objc)]

use std::mem::transmute;
use std::os::darwin::objc;

#[link(name = "Foundation", kind = "framework")]
unsafe extern "C" {}

#[link(name = "objc", kind = "dylib")]
unsafe extern "C" {
    unsafe fn objc_msgSend();
}

fn main() {
    let msg_send_fn = unsafe {
        transmute::<
            unsafe extern "C" fn(),
            unsafe extern "C" fn(objc::Class, objc::SEL) -> objc::Class,
        >(objc_msgSend)
    };
    let static_sel = objc::selector!("class");
    let static_class = objc::class!("NSObject");
    let runtime_class = unsafe { msg_send_fn(static_class, static_sel) };
    assert_eq!(static_class, runtime_class);
}
