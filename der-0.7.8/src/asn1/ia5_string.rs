//! ASN.1 `IA5String` support.

use crate::{asn1::AnyRef, FixedTag, Result, StrRef, Tag};
use core::{fmt, ops::Deref};

macro_rules! impl_ia5_string {
    ($type: ty) => {
        impl_ia5_string!($type,);
    };
    ($type: ty, $($li: lifetime)?) => {
        impl_string_type!($type, $($li),*);

        impl<$($li),*> FixedTag for $type {
            const TAG: Tag = Tag::Ia5String;
        }

        impl<$($li),*> fmt::Debug for $type {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "Ia5String({:?})", self.as_str())
            }
        }
    };
}

/// ASN.1 `IA5String` type.
///
/// Supports the [International Alphabet No. 5 (IA5)] character encoding, i.e.
/// the lower 128 characters of the ASCII alphabet. (Note: IA5 is now
/// technically known as the International Reference Alphabet or IRA as
/// specified in the ITU-T's T.50 recommendation).
///
/// For UTF-8, use [`Utf8StringRef`][`crate::asn1::Utf8StringRef`].
///
/// This is a zero-copy reference type which borrows from the input data.
///
/// [International Alphabet No. 5 (IA5)]: https://en.wikipedia.org/wiki/T.50_%28standard%29
#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord)]
pub struct Ia5StringRef<'a> {
    /// Inner value
    inner: StrRef<'a>,
}

impl<'a> Ia5StringRef<'a> {
    /// Create a new `IA5String`.
    pub fn new<T>(input: &'a T) -> Result<Self>
    where
        T: AsRef<[u8]> + ?Sized,
    {
        let input = input.as_ref();

        // Validate all characters are within IA5String's allowed set
        if input.iter().any(|&c| c > 0x7F) {
            return Err(Self::TAG.value_error());
        }

        StrRef::from_bytes(input)
            .map(|inner| Self { inner })
            .map_err(|_| Self::TAG.value_error())
    }
}

impl_ia5_string!(Ia5StringRef<'a>, 'a);

impl<'a> Deref for Ia5StringRef<'a> {
    type Target = StrRef<'a>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a> From<&Ia5StringRef<'a>> for Ia5StringRef<'a> {
    fn from(value: &Ia5StringRef<'a>) -> Ia5StringRef<'a> {
        *value
    }
}

impl<'a> From<Ia5StringRef<'a>> for AnyRef<'a> {
    fn from(internationalized_string: Ia5StringRef<'a>) -> AnyRef<'a> {
        AnyRef::from_tag_and_value(Tag::Ia5String, internationalized_string.inner.into())
    }
}

#[cfg(feature = "alloc")]
pub use self::allocation::Ia5String;

#[cfg(feature = "alloc")]
mod allocation {
    use super::Ia5StringRef;
    use crate::{
        asn1::AnyRef,
        referenced::{OwnedToRef, RefToOwned},
        Error, FixedTag, Result, StrOwned, Tag,
    };
    use alloc::string::String;
    use core::{fmt, ops::Deref};

    /// ASN.1 `IA5String` type.
    ///
    /// Supports the [International Alphabet No. 5 (IA5)] character encoding, i.e.
    /// the lower 128 characters of the ASCII alphabet. (Note: IA5 is now
    /// technically known as the International Reference Alphabet or IRA as
    /// specified in the ITU-T's T.50 recommendation).
    ///
    /// For UTF-8, use [`String`][`alloc::string::String`].
    ///
    /// [International Alphabet No. 5 (IA5)]: https://en.wikipedia.org/wiki/T.50_%28standard%29
    #[derive(Clone, Eq, PartialEq, PartialOrd, Ord)]
    pub struct Ia5String {
        /// Inner value
        inner: StrOwned,
    }

    impl Ia5String {
        /// Create a new `IA5String`.
        pub fn new<T>(input: &T) -> Result<Self>
        where
            T: AsRef<[u8]> + ?Sized,
        {
            let input = input.as_ref();
            Ia5StringRef::new(input)?;

            StrOwned::from_bytes(input)
                .map(|inner| Self { inner })
                .map_err(|_| Self::TAG.value_error())
        }
    }

    impl_ia5_string!(Ia5String);

    impl Deref for Ia5String {
        type Target = StrOwned;

        fn deref(&self) -> &Self::Target {
            &self.inner
        }
    }

    impl<'a> From<Ia5StringRef<'a>> for Ia5String {
        fn from(international_string: Ia5StringRef<'a>) -> Ia5String {
            let inner = international_string.inner.into();
            Self { inner }
        }
    }

    impl<'a> From<&'a Ia5String> for AnyRef<'a> {
        fn from(international_string: &'a Ia5String) -> AnyRef<'a> {
            AnyRef::from_tag_and_value(Tag::Ia5String, (&international_string.inner).into())
        }
    }

    impl<'a> RefToOwned<'a> for Ia5StringRef<'a> {
        type Owned = Ia5String;
        fn ref_to_owned(&self) -> Self::Owned {
            Ia5String {
                inner: self.inner.ref_to_owned(),
            }
        }
    }

    impl OwnedToRef for Ia5String {
        type Borrowed<'a> = Ia5StringRef<'a>;
        fn owned_to_ref(&self) -> Self::Borrowed<'_> {
            Ia5StringRef {
                inner: self.inner.owned_to_ref(),
            }
        }
    }

    impl TryFrom<String> for Ia5String {
        type Error = Error;

        fn try_from(input: String) -> Result<Self> {
            Ia5StringRef::new(&input)?;

            StrOwned::new(input)
                .map(|inner| Self { inner })
                .map_err(|_| Self::TAG.value_error())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Ia5StringRef;
    use crate::Decode;
    use hex_literal::hex;

    #[test]
    fn parse_bytes() {
        let example_bytes = hex!("16 0d 74 65 73 74 31 40 72 73 61 2e 63 6f 6d");
        let internationalized_string = Ia5StringRef::from_der(&example_bytes).unwrap();
        assert_eq!(internationalized_string.as_str(), "test1@rsa.com");
    }
}
