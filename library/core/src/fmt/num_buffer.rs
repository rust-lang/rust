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
            #[stable(feature = "int_format_into", since = "1.98.0")]
            impl NumBufferTrait for $signed {
                // `+ 2` and not `+ 1` to include the `-` character.
                const DEFAULT: Self::Buf = [MaybeUninit::<u8>::uninit(); $signed::MAX.ilog10() as usize + 2];
                type Buf = [MaybeUninit<u8>; $signed::MAX.ilog10() as usize + 2];
            }
            #[stable(feature = "int_format_into", since = "1.98.0")]
            impl NumBufferTrait for $unsigned {
                const DEFAULT: Self::Buf = [MaybeUninit::<u8>::uninit(); $unsigned::MAX.ilog10() as usize + 1];
                type Buf = [MaybeUninit<u8>; $unsigned::MAX.ilog10() as usize + 1];
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
#[stable(feature = "int_format_into", since = "1.98.0")]
#[repr(transparent)]
pub struct NumBuffer<T: NumBufferTrait> {
    pub(crate) buf: T::Buf,
    phantom: core::marker::PhantomData<T>,
}

#[stable(feature = "int_format_into", since = "1.98.0")]
impl<T: NumBufferTrait> core::fmt::Debug for NumBuffer<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NumBuffer").finish()
    }
}

#[stable(feature = "int_format_into", since = "1.98.0")]
impl<T: NumBufferTrait> NumBuffer<T> {
    /// Initializes internal buffer.
    #[stable(feature = "int_format_into", since = "1.98.0")]
    #[rustc_const_stable(feature = "int_format_into", since = "1.98.0")]
    pub const fn new() -> Self {
        NumBuffer { buf: T::DEFAULT, phantom: core::marker::PhantomData }
    }
}

impl<T: NumBufferTrait> NumBuffer<T> {
    /// Allows to cast between `NumBuffer` types at compile-time without new allocation.
    ///
    /// # Examples
    ///
    /// ```
    /// use core::fmt::NumBuffer;
    ///
    /// let mut buf = NumBuffer::<u32>::new();
    ///
    /// assert_eq!(16u16.format_into(buf.cast_into::<u16>()), "16");
    /// assert_eq!(u16::MAX.format_into(buf.cast_into::<u16>()), u16::MAX.to_string());
    ///
    /// assert_eq!(-16i16.format_into(buf.cast_into::<i16>()), "-16");
    /// assert_eq!(i16::MIN.format_into(buf.cast_into::<i16>()), i16::MIN.to_string());
    /// ```
    ///
    /// If you try to cast to a `NumBuffer` with a bigger buffer size, it will not compile:
    ///
    /// ```compile_fail
    /// use core::fmt::NumBuffer;
    ///
    /// let mut buf = NumBuffer::<u32>::new();
    /// // Cannot work since `i32` requires a bigger buffer (because of the `-` sign).
    /// let buf = buf.cast_into::<i32>();
    /// ```
    #[unstable(feature = "fmt_internals", issue = "none")]
    #[rustc_const_unstable(feature = "fmt_internals", issue = "none")]
    #[track_caller]
    pub const fn cast_into<U: NumBufferTrait>(&mut self) -> &mut NumBuffer<U> {
        const {
            assert!(
                core::mem::size_of::<T::Buf>() >= core::mem::size_of::<U::Buf>(),
                "target `NumBuffer` size must be smaller or equal to source `NumBuffer` size"
            );
        }
        // SAFETY: The target `NumBuffer` buffer is not bigger so this conversion is ok.
        unsafe { core::mem::transmute::<&mut NumBuffer<T>, &mut NumBuffer<U>>(self) }
    }
}
