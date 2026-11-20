#![warn(clippy::too_many_arguments)]

fn not_too_many(p1: u8, p2: u8, p3: u8, p4: u8, p5: u8, p6: u8, p7: u8, p8: u8, p9: u8, p10: u8) {}
fn too_many(p1: u8, p2: u8, p3: u8, p4: u8, p5: u8, p6: u8, p7: u8, p8: u8, p9: u8, p10: u8, p11: u8) {}
//~^ ERROR: this function has too many arguments

fn main() {}
