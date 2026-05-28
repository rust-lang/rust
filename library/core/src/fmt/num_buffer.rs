use crate::mem::MaybeUninit;

/// Trait used to describe the maximum number of digits in decimal base of the implemented integer.
#[unstable(feature = "fmt_internals", issue = "none")]
pub trait NumBufferTrait {
    /// Used for initializing the `NumberBuffer` value.
    #[unstable(feature = "fmt_internals", issue = "none")]
    const DEFAULT: Self::Buf;
    /// The actual underlying type.
    #[unstable(feature = "fmt_internals", issue = "none")]
    type Buf: AsRef<[MaybeUninit<u8>]> + AsMut<[MaybeUninit<u8>]>;
}

macro_rules! impl_NumBufferTrait {
    ($($signed:ident, $unsigned:ident,)*) => {
        $(
            #[stable(feature = "int_format_into", since = "CURRENT_RUSTC_VERSION")]
            impl NumBufferTrait for $signed {
                // `+ 2` and not `+ 1` to include the `-` character.
                const DEFAULT: Self::Buf = [MaybeUninit::<u8>::uninit(); $signed::MAX.ilog(10) as usize + 2];
                type Buf = [MaybeUninit<u8>; $signed::MAX.ilog(10) as usize + 2];
            }
            #[stable(feature = "int_format_into", since = "CURRENT_RUSTC_VERSION")]
            impl NumBufferTrait for $unsigned {
                const DEFAULT: Self::Buf = [MaybeUninit::<u8>::uninit(); $unsigned::MAX.ilog(10) as usize + 1];
                type Buf = [MaybeUninit<u8>; $unsigned::MAX.ilog(10) as usize + 1];
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
///
/// # Examples
///
/// ```
/// use core::fmt::NumBuffer;
///
/// let mut buf = NumBuffer::new();
/// let n1 = 1972u32;
/// assert_eq!(n1.format_into(&mut buf), "1972");
///
/// // Formatting a negative integer includes the sign.
/// let mut buf = NumBuffer::new();
/// let n2 = -1972i32;
/// assert_eq!(n2.format_into(&mut buf), "-1972");
/// ```
#[stable(feature = "int_format_into", since = "CURRENT_RUSTC_VERSION")]
pub struct NumBuffer<T: NumBufferTrait> {
    pub(crate) buf: T::Buf,
    phantom: core::marker::PhantomData<T>,
}

#[stable(feature = "int_format_into", since = "CURRENT_RUSTC_VERSION")]
impl<T: NumBufferTrait> core::fmt::Debug for NumBuffer<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NumBuffer").finish()
    }
}

#[stable(feature = "int_format_into", since = "CURRENT_RUSTC_VERSION")]
impl<T: NumBufferTrait> NumBuffer<T> {
    /// Initializes internal buffer.
    #[stable(feature = "int_format_into", since = "CURRENT_RUSTC_VERSION")]
    #[rustc_const_stable(feature = "int_format_into", since = "CURRENT_RUSTC_VERSION")]
    pub const fn new() -> Self {
        // FIXME: Once const generics feature is working, use `T::BUF_SIZE` instead of 40.
        NumBuffer { buf: T::DEFAULT, phantom: core::marker::PhantomData }
    }
}
