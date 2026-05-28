pub fn foo(_const: *const f32, _mut: *mut i64) {}

fn main() {
    let _ = 0 as *const usize;
    //~^ zero_ptr
    let _ = 0 as *mut f64;
    //~^ zero_ptr
    let _: *const u8 = 0 as *const _;
    //~^ zero_ptr

    foo(0 as _, 0 as _);
    foo(0 as *const _, 0 as *mut _);
    //~^ zero_ptr
    //~| zero_ptr

    let z = 0;
    let _ = z as *const usize; // this is currently not caught
}

const fn in_const_context() {
    #[clippy::msrv = "1.23"]
    let _: *const usize = 0 as *const _;
    #[clippy::msrv = "1.24"]
    let _: *const usize = 0 as *const _;
    //~^ zero_ptr
}
