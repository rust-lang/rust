pub trait IpuItertools {
    fn ipu_flatten(&self) -> u32 {
        1
    }
}

impl IpuItertools for char {}
