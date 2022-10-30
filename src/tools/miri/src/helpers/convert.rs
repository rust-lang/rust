use implementations::NarrowerThan;

/// Replacement for `as` casts going from wide integer to narrower integer.
///
/// # Example
///
/// ```ignore
/// let x = 99_u64;
/// let lo = x.truncate::<u16>();
/// // lo is of type u16, equivalent to `x as u16`.
/// ```
pub(crate) trait Truncate: Sized {
    fn truncate<To>(self) -> To
    where
        To: NarrowerThan<Self>,
    {
        NarrowerThan::truncate_from(self)
    }
}

impl Truncate for u16 {}
impl Truncate for u32 {}
impl Truncate for u64 {}
impl Truncate for u128 {}

mod implementations {
    pub(crate) trait NarrowerThan<T> {
        fn truncate_from(wide: T) -> Self;
    }

    macro_rules! impl_narrower_than {
        ($(NarrowerThan<{$($ty:ty),*}> for $self:ty)*) => {
            $($(
                impl NarrowerThan<$ty> for $self {
                    fn truncate_from(wide: $ty) -> Self {
                        wide as Self
                    }
                }
            )*)*
        };
    }

    impl_narrower_than! {
        NarrowerThan<{u128, u64, u32, u16}> for u8
        NarrowerThan<{u128, u64, u32}> for u16
        NarrowerThan<{u128, u64}> for u32
        NarrowerThan<{u128}> for u64
    }
}
