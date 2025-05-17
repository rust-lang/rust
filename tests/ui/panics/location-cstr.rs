//@ run-pass
//@ compile-flags: -Zlocation-detail=line,file,cstr

use std::panic::Location;

fn main() {
    let location = Location::caller();
    let len = location.file().len();
    let ptr = location.file().as_ptr();

    let zero_terminator = unsafe { core::ptr::read(ptr.add(len)) };
    assert_eq!(zero_terminator, 0u8);
}
