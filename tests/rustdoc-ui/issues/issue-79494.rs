//@ only-64bit

pub const ZST: &[u8] = unsafe { std::mem::transmute(1usize) };
//~^ ERROR: cannot transmute between types of different sizes, or dependently-sized types [E0512]
