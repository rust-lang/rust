trait Trait {}

// Make sure casts between thin-pointer <-> fat pointer obey RFC401
fn main() {
    let a: &[i32] = &[1, 2, 3];
    let b: Box<[i32]> = Box::new([1, 2, 3]);
    let p = a as *const [i32];
    let q = a.as_ptr();

    a as usize; //~ ERROR casting
    a as isize; //~ ERROR casting
    a as i16; //~ ERROR casting `&[i32]` as `i16` is invalid
    a as u32; //~ ERROR casting `&[i32]` as `u32` is invalid
    b as usize; //~ ERROR non-primitive cast
    p as usize;
    //~^ ERROR casting

    // #22955
    q as *const [i32]; //~ ERROR cannot cast

    // #21397
    let t: *mut (dyn Trait + 'static) = 0 as *mut _; //~ ERROR casting
    let mut fail: *const str = 0 as *const str; //~ ERROR casting
}
