// compile-flags: -Z print-type-sizes
// compile-pass

#![feature(never_type)]
#![feature(start)]

#[start]
fn start(_: isize, _: *const *const u8) -> isize {
    let _x: Option<!> = None;
    let _y: Result<u32, !> = Ok(42);
    0
}
