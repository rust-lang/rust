fn main() {
    let _val = unsafe { std::mem::MaybeUninit::<*const u8>::uninit().assume_init() };
    //~^ ERROR: uninitialized
}
