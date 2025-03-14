/// XorShiftRng
pub(crate) struct DeterministicRng {
    count: usize,
    x: u32,
    y: u32,
    z: u32,
    w: u32,
}

impl DeterministicRng {
    pub(crate) fn new() -> Self {
        DeterministicRng { count: 0, x: 0x193a6754, y: 0xa8a7d469, z: 0x97830e05, w: 0x113ba7bb }
    }

    /// Guarantees that each returned number is unique.
    pub(crate) fn next(&mut self) -> u32 {
        self.count += 1;
        assert!(self.count <= 70029);
        let x = self.x;
        let t = x ^ (x << 11);
        self.x = self.y;
        self.y = self.z;
        self.z = self.w;
        let w_ = self.w;
        self.w = w_ ^ (w_ >> 19) ^ (t ^ (t >> 8));
        self.w
    }
}
