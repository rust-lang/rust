use std::hint::black_box;

fn private_target(x: u32) -> u32 {
    x.wrapping_add(1)
}

pub fn call_target(x: u32) -> u32 {
    let f: fn(u32) -> u32 = black_box(private_target);
    f(x)
}
