#![derive(Clone, Copy)] //~ ERROR `derive` attribute cannot be used at crate level

use std::ptr::addr_of;

const UNINHABITED_VARIANT: () = unsafe {
    let v = *addr_of!(data).cast();
};

fn main() {}
