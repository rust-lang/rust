#![feature(const_raw_ptr_to_usize_cast)]

fn main() {
    const OK: usize = unsafe { 0 as *const i32 as usize };

    const _ERROR: usize = unsafe { &0 as *const i32 as usize };
    //~^ ERROR [const_err]
    //~| NOTE cannot cast pointer to integer because it was not created by cast from integer
    //~| NOTE
    //~| NOTE `#[deny(const_err)]` on by default
    //~| WARN this was previously accepted by the compiler but is being phased out
    //~| NOTE see issue #71800
}
