// 480
fn f(_: (u8, (u8, (u8, (u8, (u8, (u8,))))))) {}
// 550
fn f2(_: (u8, (u8, (u8, (u8, (u8, (u8, u8))))))) {}
//~^ ERROR: very complex type used

fn main() {}
