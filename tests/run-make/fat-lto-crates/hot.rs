#[cfg(not(hot_b))]
pub static TABLE: [u32; 8] = [3, 1, 4, 1, 5, 9, 2, 6];
#[cfg(hot_b)]
pub static TABLE: [u32; 8] = [3, 1, 4, 1, 5, 9, 2, 7];

pub trait Op {
    fn apply(&self, x: u32) -> u32;
}

pub struct Add(pub u32);

impl Op for Add {
    fn apply(&self, x: u32) -> u32 {
        x.wrapping_add(self.0).wrapping_add(TABLE[(x % 8) as usize])
    }
}

#[cfg(not(hot_b))]
pub fn twist(x: u32) -> u32 {
    (x ^ 0x5a5a).rotate_left(TABLE[(x % 8) as usize])
}

#[cfg(hot_b)]
pub fn twist(x: u32) -> u32 {
    (x ^ 0x6b6b).rotate_left(TABLE[(x % 8) as usize])
}
