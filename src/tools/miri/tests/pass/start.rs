#![feature(start)]

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    println!("Hello from start!");

    0
}
