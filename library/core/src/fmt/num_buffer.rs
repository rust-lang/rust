use crate::mem::MaybeUninit;

/// Trait used to describe the maximum number of digits in decimal base of the implemented integer.
#[unstable(feature = "fmt_internals", issue = "none")]
pub trait NumBufferTrait {
    /// Maximum number of digits in decimal base of the implemented integer.
    #[unstable(feature = "fmt_internals", issue = "none")]
    const BUF_SIZE: usize;
}

macro_rules! impl_NumBufferTrait {
    ($($signed:ident, $unsigned:ident,)*) => {
        $(
            #[stable(feature = "int_format_into", since = "CURRENT_RUSTC_VERSION")]
            impl NumBufferTrait for $signed {
                // `+ 2` and not `+ 1` to include the `-` character.
                const BUF_SIZE: usize = $signed::MAX.ilog(10) as usize + 2;
            }
            #[stable(feature = "int_format_into", since = "CURRENT_RUSTC_VERSION")]
            impl NumBufferTrait for $unsigned {
                const BUF_SIZE: usize = $unsigned::MAX.ilog(10) as usize + 1;
            }
        )*
    }
}

impl_NumBufferTrait! {
    i8, u8,
    i16, u16,
    i32, u32,
    i64, u64,
    isize, usize,
    i128, u128,
}

/// A buffer wrapper of which the internal size is based on the maximum
/// number of digits the associated integer can have.
#[stable(feature = "int_format_into", since = "CURRENT_RUSTC_VERSION")]
#[derive(Debug)]
pub struct NumBuffer<T: NumBufferTrait> {
    // FIXME: Once const generics feature is working, use `T::BUF_SIZE` instead of 40.
    pub(crate) buf: [MaybeUninit<u8>; 40],
    // FIXME: Remove this field once we can actually use `T`.
    phantom: core::marker::PhantomData<T>,
}

#[stable(feature = "int_format_into", since = "CURRENT_RUSTC_VERSION")]
impl<T: NumBufferTrait> NumBuffer<T> {
    /// Initializes internal buffer.
    #[stable(feature = "int_format_into", since = "CURRENT_RUSTC_VERSION")]
    #[rustc_const_stable(feature = "int_format_into", since = "CURRENT_RUSTC_VERSION")]
    pub const fn new() -> Self {
        // FIXME: Once const generics feature is working, use `T::BUF_SIZE` instead of 40.
        NumBuffer { buf: [MaybeUninit::<u8>::uninit(); 40], phantom: core::marker::PhantomData }
    }

    pub(crate) const fn capacity(&self) -> usize {
        self.buf.len()
    }
}
