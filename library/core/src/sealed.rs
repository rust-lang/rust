/// This trait being unreachable from outside the crate
/// prevents outside implementations of our extension traits.
/// This allows adding more trait methods in the future.
#[unstable(feature = "core_sealed", issue = "none")]
pub trait Sealed {}

macro_rules! integers_sealed_impl {
    ($($T:ty)+) => {
        $(
            #[unstable(feature = "core_sealed", issue = "none")]
            impl Sealed for $T {}

            #[unstable(feature = "core_sealed", issue = "none")]
            impl Sealed for &'_ $T {}
        )+
    };
}

// Used by sealed `WrappingAdd` trait.
integers_sealed_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }
