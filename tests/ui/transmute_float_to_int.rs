#![feature(const_fn_transmute)]
#![warn(clippy::transmute_float_to_int)]

fn float_to_int() {
    let _: u32 = unsafe { std::mem::transmute(1f32) };
    let _: i32 = unsafe { std::mem::transmute(1f32) };
    let _: u64 = unsafe { std::mem::transmute(1f64) };
    let _: i64 = unsafe { std::mem::transmute(1f64) };
    let _: u64 = unsafe { std::mem::transmute(1.0) };
    let _: u64 = unsafe { std::mem::transmute(-1.0) };
}

mod issue_5747 {
    const VALUE: u32 = unsafe { std::mem::transmute(1f32) };

    const fn to_bits(v: f32) -> u32 {
        unsafe { std::mem::transmute(v) }
    }
}

fn main() {}
