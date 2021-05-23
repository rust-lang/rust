// revisions: mir thir
// [thir]compile-flags: -Z thir-unsafeck

#![feature(const_raw_ptr_to_usize_cast)]

fn main() {
    const Y: u32 = 0;
    // Cast in `const` without `unsafe` block
    const SAFE: usize = {
        &Y as *const u32 as usize
        //~^ ERROR cast of pointer to int is unsafe and requires unsafe
    };
}

// Cast in `const fn` without `unsafe` block
const fn test() -> usize {
    &0 as *const i32 as usize
    //~^ ERROR cast of pointer to int is unsafe and requires unsafe
}
