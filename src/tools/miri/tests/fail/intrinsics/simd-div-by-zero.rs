#![feature(platform_intrinsics, repr_simd)]

extern "platform-intrinsic" {
    pub(crate) fn simd_div<T>(x: T, y: T) -> T;
}

#[repr(simd)]
#[allow(non_camel_case_types)]
struct i32x2(i32, i32);

fn main() {
    unsafe {
        let x = i32x2(1, 1);
        let y = i32x2(1, 0);
        simd_div(x, y); //~ERROR: Undefined Behavior: dividing by zero
    }
}
