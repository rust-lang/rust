fn main() {
    use std::ptr;

    const DATA: [u32; 1] = [42];

    const PAST_END_PTR: *const u32 = unsafe { DATA.as_ptr().add(1) };

    const _READ: u32 = unsafe { ptr::read(PAST_END_PTR) };
    //~^ ERROR evaluation of constant value failed
    const _CONST_READ: u32 = unsafe { PAST_END_PTR.read() };
    //~^ ERROR evaluation of constant value failed
    const _MUT_READ: u32 = unsafe { (PAST_END_PTR as *mut u32).read() };
    //~^ ERROR evaluation of constant value failed
}
