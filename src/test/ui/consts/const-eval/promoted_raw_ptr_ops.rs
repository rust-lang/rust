#![feature(const_raw_ptr_to_usize_cast, const_compare_raw_pointers, const_raw_ptr_deref)]

fn main() {
    let x: &'static bool = &(42 as *const i32 == 43 as *const i32);
    //~^ ERROR does not live long enough
    let y: &'static usize = &(&1 as *const i32 as usize + 1); //~ ERROR does not live long enough
    let z: &'static i32 = &(unsafe { *(42 as *const i32) }); //~ ERROR does not live long enough
    let a: &'static bool = &(main as fn() == main as fn()); //~ ERROR does not live long enough
}
