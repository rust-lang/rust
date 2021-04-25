// gate-test-const_raw_ptr_to_usize_cast
// revisions: with_feature without_feature

#![cfg_attr(with_feature, feature(const_raw_ptr_to_usize_cast))]

fn main() {
    const X: usize = unsafe {
        main as usize //[without_feature]~ ERROR casting pointers to integers in constants is unstable
    };
    const Y: u32 = 0;
    const Z: usize = unsafe {
        &Y as *const u32 as usize //[without_feature]~ ERROR is unstable
    };
    // Cast in `const` without `unsafe` block
    const SAFE: usize = {
        &Y as *const u32 as usize //[without_feature]~ ERROR is unstable
        //[with_feature]~^ ERROR cast of pointer to int is unsafe and requires unsafe
    };
}

// Cast in `const fn` without `unsafe` block
const fn test() -> usize {
    &0 as *const i32 as usize //[without_feature]~ ERROR is unstable
    //[with_feature]~^ ERROR cast of pointer to int is unsafe and requires unsafe
}
