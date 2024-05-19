#![no_std]
#![feature(start)]

extern crate std;

#[start]
fn start(_: isize, _: *const *const u8) -> isize where (): Copy { //~ ERROR [E0647]
    0
}
