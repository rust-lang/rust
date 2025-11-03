#![feature(extern_types)]

// Regression test for #79409

extern "C" {
    type Device;
}

unsafe fn make_device() -> Box<Device> {
//~^ ERROR the size for values of type `Device` cannot be known
    Box::from_raw(0 as *mut _)
//~^ ERROR the size for values of type `Device` cannot be known
//~| ERROR the size for values of type `Device` cannot be known
}

fn main() {
    let d: Device = unsafe { *make_device() };
//~^ ERROR the size for values of type `Device` cannot be known
//~| ERROR the size for values of type `Device` cannot be known
}
