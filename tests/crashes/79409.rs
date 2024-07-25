//@ known-bug: #79409
//@ compile-flags: -Cdebuginfo=2

#![feature(extern_types)]
#![feature(unsized_locals)]

extern {
    type Device;
}

unsafe fn make_device() -> Box<Device> {
    Box::from_raw(0 as *mut _)
}

fn main() {
    let d: Device = unsafe { *make_device() };
}
