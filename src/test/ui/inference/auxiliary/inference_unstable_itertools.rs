pub trait IpuItertools {
    fn ipu_flatten(&self) -> u32 {
        1
    }

    const C: i32;
}

impl IpuItertools for char {
    const C: i32 = 1;
}
