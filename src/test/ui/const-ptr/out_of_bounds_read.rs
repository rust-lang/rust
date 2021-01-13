// error-pattern: any use of this value will cause an error

#![feature(const_ptr_read)]
#![feature(const_ptr_offset)]

fn main() {
    use std::ptr;

    const DATA: [u32; 1] = [42];

    const PAST_END_PTR: *const u32 = unsafe { DATA.as_ptr().add(1) };

    const _READ: u32 = unsafe { ptr::read(PAST_END_PTR) };
    const _CONST_READ: u32 = unsafe { PAST_END_PTR.read() };
    const _MUT_READ: u32 = unsafe { (PAST_END_PTR as *mut u32).read() };
}
