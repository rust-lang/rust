use std as line;

const C: u32 = line!(); //~ ERROR cannot determine resolution for the macro `line`

fn main() {}
