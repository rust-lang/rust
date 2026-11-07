pub trait FlushSubnormals: Sized {
    fn flush(self) -> Self {
        self
    }
}

impl<T> FlushSubnormals for *const T {}
impl<T> FlushSubnormals for *mut T {}

impl FlushSubnormals for i8 {}
impl FlushSubnormals for i16 {}
impl FlushSubnormals for i32 {}
impl FlushSubnormals for i64 {}
impl FlushSubnormals for isize {}

impl FlushSubnormals for u8 {}
impl FlushSubnormals for u16 {}
impl FlushSubnormals for u32 {}
impl FlushSubnormals for u64 {}
impl FlushSubnormals for usize {}

impl FlushSubnormals for f16 {}
impl FlushSubnormals for f64 {}

impl FlushSubnormals for f32 {
    fn flush(self) -> Self {
        // Correct for non-IEEE behavior for f32 on arm and powerpc.
        let ppc_flush = cfg!(all(
            any(
                target_arch = "powerpc",
                all(target_arch = "powerpc64", target_endian = "big")
            ),
            target_feature = "altivec",
            not(target_feature = "vsx"),
        ));
        let arm_flush = cfg!(all(target_arch = "arm", target_feature = "neon"));
        let flush = ppc_flush || arm_flush;

        if flush && self.is_subnormal() {
            f32::copysign(0.0, self)
        } else {
            self
        }
    }
}

/// NOTE: altivec had a subnormal flushing bug in older QEMU versions.
/// <https://gitlab.com/qemu-project/qemu/-/issues/1779>
pub fn flush_in<T: FlushSubnormals>(x: T) -> T {
    x.flush()
}

pub fn flush<T: FlushSubnormals>(x: T) -> T {
    x.flush()
}
