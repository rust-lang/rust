#[warn(clippy::transmute_float_to_int)]

fn float_to_int() {
    let _: u32 = unsafe { std::mem::transmute(1f32) };
    let _: i32 = unsafe { std::mem::transmute(1f32) };
    let _: u64 = unsafe { std::mem::transmute(1f64) };
    let _: i64 = unsafe { std::mem::transmute(1f64) };
    let _: u64 = unsafe { std::mem::transmute(1.0) };
    let _: u64 = unsafe { std::mem::transmute(-1.0) };
}

fn main() {}
