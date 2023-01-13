// Used to cause ICE

static VEC: [u32; 256] = vec![];
//~^ ERROR mismatched types

fn main() {}
