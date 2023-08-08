//@run-rustfix
pub fn foo(_const: *const f32, _mut: *mut i64) {}

fn main() {
    let _ = 0 as *const usize;
    let _ = 0 as *mut f64;
    let _: *const u8 = 0 as *const _;

    foo(0 as _, 0 as _);
    foo(0 as *const _, 0 as *mut _);

    let z = 0;
    let _ = z as *const usize; // this is currently not caught
}
