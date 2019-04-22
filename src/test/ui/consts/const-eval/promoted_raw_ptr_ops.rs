#![feature(const_raw_ptr_to_usize_cast, const_compare_raw_pointers, const_raw_ptr_deref)]

fn main() {
    let x: &'static bool = &(42 as *const i32 == 43 as *const i32);
    //~^ ERROR temporary value dropped while borrowed
    let y: &'static usize = &(&1 as *const i32 as usize + 1);
    //~^ ERROR temporary value dropped while borrowed
    let z: &'static i32 = &(unsafe { *(42 as *const i32) });
    //~^ ERROR temporary value dropped while borrowed
    let a: &'static bool = &(main as fn() == main as fn());
    //~^ ERROR temporary value dropped while borrowed
}
