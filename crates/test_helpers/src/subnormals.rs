pub trait FlushSubnormals: Sized {
    fn flush(self) -> Self {
        self
    }
}

impl<T> FlushSubnormals for *const T {}
impl<T> FlushSubnormals for *mut T {}

macro_rules! impl_float {
    { $($ty:ty),* } => {
        $(
        impl FlushSubnormals for $ty {
            fn flush(self) -> Self {
                let is_f32 = core::mem::size_of::<Self>() == 4;
                let ppc_flush = is_f32 && cfg!(all(target_arch = "powerpc64", target_endian = "big", not(target_feature = "vsx")));
                let arm_flush = is_f32 && cfg!(all(target_arch = "arm", target_feature = "neon"));
                let flush = ppc_flush || arm_flush;
                if flush && self.is_subnormal() {
                    <$ty>::copysign(0., self)
                } else {
                    self
                }
            }
        }
        )*
    }
}

macro_rules! impl_else {
    { $($ty:ty),* } => {
        $(
        impl FlushSubnormals for $ty {}
        )*
    }
}

impl_float! { f32, f64 }
impl_else! { i8, i16, i32, i64, isize, u8, u16, u32, u64, usize }
