#![feature(extern_types)]

// Regression test for #79409

extern "C" {
    type Device;
}

unsafe fn make_device() -> Box<Device> {
    Box::from_raw(0 as *mut _)
}

fn main() {
    let d: Device = unsafe { *make_device() };
//~^ ERROR the size for values of type `Device` cannot be known
}
