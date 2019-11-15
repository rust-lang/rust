#[warn(clippy::as_conversions)]

fn main() {
    let i = 0u32 as u64;

    let j = &i as *const u64 as *mut u64;
}
