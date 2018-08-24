#![feature(start)]

#[start]
fn start(argc: isize, argv: *const *const u8, crate_map: *const u8) -> isize {
    //~^ start function has wrong type
   0
}
