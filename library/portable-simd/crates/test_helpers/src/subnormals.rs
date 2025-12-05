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
                let is_f32 = size_of::<Self>() == 4;
                let ppc_flush = is_f32 && cfg!(all(
                    any(target_arch = "powerpc", all(target_arch = "powerpc64", target_endian = "big")),
                    target_feature = "altivec",
                    not(target_feature = "vsx"),
                ));
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

/// AltiVec should flush subnormal inputs to zero, but QEMU seems to only flush outputs.
/// https://gitlab.com/qemu-project/qemu/-/issues/1779
#[cfg(all(
    any(target_arch = "powerpc", target_arch = "powerpc64"),
    target_feature = "altivec"
))]
fn in_buggy_qemu() -> bool {
    use std::sync::OnceLock;
    static BUGGY: OnceLock<bool> = OnceLock::new();

    fn add(x: f32, y: f32) -> f32 {
        #[cfg(target_arch = "powerpc")]
        use core::arch::powerpc::*;
        #[cfg(target_arch = "powerpc64")]
        use core::arch::powerpc64::*;

        let array: [f32; 4] =
            unsafe { core::mem::transmute(vec_add(vec_splats(x), vec_splats(y))) };
        array[0]
    }

    *BUGGY.get_or_init(|| add(-1.0857398e-38, 0.).is_sign_negative())
}

#[cfg(all(
    any(target_arch = "powerpc", target_arch = "powerpc64"),
    target_feature = "altivec"
))]
pub fn flush_in<T: FlushSubnormals>(x: T) -> T {
    if in_buggy_qemu() {
        x
    } else {
        x.flush()
    }
}

#[cfg(not(all(
    any(target_arch = "powerpc", target_arch = "powerpc64"),
    target_feature = "altivec"
)))]
pub fn flush_in<T: FlushSubnormals>(x: T) -> T {
    x.flush()
}

pub fn flush<T: FlushSubnormals>(x: T) -> T {
    x.flush()
}
