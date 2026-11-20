//@ revisions: SIZE SIZEMIN
//@[SIZE] compile-flags: -Copt-level=s
//@[SIZEMIN] compile-flags: -Copt-level=z

#[inline(never)]
#[no_mangle]
pub fn aux_quiet(x: u32) -> u32 {
    x.wrapping_add(1)
}

#[inline(never)]
#[no_mangle]
pub fn aux_panicky(x: u32) -> u32 {
    if x == 0 {
        panic!("zero");
    }
    x
}
