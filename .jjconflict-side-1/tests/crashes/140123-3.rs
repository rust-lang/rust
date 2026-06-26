//@ known-bug: #140123
//@ compile-flags: --crate-type lib

const ICE: [&mut [()]; 2] = [const { empty_mut() }; 2];

const fn empty_mut() -> &'static mut [()] {
    unsafe {
        std::slice::from_raw_parts_mut(std::ptr::dangling_mut(), 0)
    }
}
