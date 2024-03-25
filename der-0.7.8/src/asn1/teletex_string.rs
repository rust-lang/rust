//! ASN.1 `TeletexString` support.
//!
use crate::{asn1::AnyRef, FixedTag, Result, StrRef, Tag};
use core::{fmt, ops::Deref};

macro_rules! impl_teletex_string {
    ($type: ty) => {
        impl_teletex_string!($type,);
    };
    ($type: ty, $($li: lifetime)?) => {
        impl_string_type!($type, $($li),*);

        impl<$($li),*> FixedTag for $type {
            const TAG: Tag = Tag::TeletexString;
        }

        impl<$($li),*> fmt::Debug for $type {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "TeletexString({:?})", self.as_str())
            }
        }
    };
}

/// ASN.1 `TeletexString` type.
///
/// Supports a subset the ASCII character set (described below).
///
/// For UTF-8, use [`Utf8StringRef`][`crate::asn1::Utf8StringRef`] instead.
/// For the full ASCII character set, use
/// [`Ia5StringRef`][`crate::asn1::Ia5StringRef`].
///
/// This is a zero-copy reference type which borrows from the input data.
///
/// # Supported characters
///
/// The standard defines a complex character set allowed in this type. However, quoting the ASN.1
/// mailing list, "a sizable volume of software in the world treats TeletexString (T61String) as a
/// simple 8-bit string with mostly Windows Latin 1 (superset of iso-8859-1) encoding".
///
#[derive(Copy, Clone, Eq, PartialEq, PartialOrd, Ord)]
pub struct TeletexStringRef<'a> {
    /// Inner value
    inner: StrRef<'a>,
}

impl<'a> TeletexStringRef<'a> {
    /// Create a new ASN.1 `TeletexString`.
    pub fn new<T>(input: &'a T) -> Result<Self>
    where
        T: AsRef<[u8]> + ?Sized,
    {
        let input = input.as_ref();

        // FIXME: support higher part of the charset
        if input.iter().any(|&c| c > 0x7F) {
            return Err(Self::TAG.value_error());
        }

        StrRef::from_bytes(input)
            .map(|inner| Self { inner })
            .map_err(|_| Self::TAG.value_error())
    }
}

impl_teletex_string!(TeletexStringRef<'a>, 'a);

impl<'a> Deref for TeletexStringRef<'a> {
    type Target = StrRef<'a>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a> From<&TeletexStringRef<'a>> for TeletexStringRef<'a> {
    fn from(value: &TeletexStringRef<'a>) -> TeletexStringRef<'a> {
        *value
    }
}

impl<'a> From<TeletexStringRef<'a>> for AnyRef<'a> {
    fn from(teletex_string: TeletexStringRef<'a>) -> AnyRef<'a> {
        AnyRef::from_tag_and_value(Tag::TeletexString, teletex_string.inner.into())
    }
}

#[cfg(feature = "alloc")]
pub use self::allocation::TeletexString;

#[cfg(feature = "alloc")]
mod allocation {
    use super::TeletexStringRef;

    use crate::{
        asn1::AnyRef,
        referenced::{OwnedToRef, RefToOwned},
        BytesRef, Error, FixedTag, Result, StrOwned, Tag,
    };
    use alloc::string::String;
    use core::{fmt, ops::Deref};

    /// ASN.1 `TeletexString` type.
    ///
    /// Supports a subset the ASCII character set (described below).
    ///
    /// For UTF-8, use [`Utf8StringRef`][`crate::asn1::Utf8StringRef`] instead.
    /// For the full ASCII character set, use
    /// [`Ia5StringRef`][`crate::asn1::Ia5StringRef`].
    ///
    /// # Supported characters
    ///
    /// The standard defines a complex character set allowed in this type. However, quoting the ASN.1
    /// mailing list, "a sizable volume of software in the world treats TeletexString (T61String) as a
    /// simple 8-bit string with mostly Windows Latin 1 (superset of iso-8859-1) encoding".
    ///
    #[derive(Clone, Eq, PartialEq, PartialOrd, Ord)]
    pub struct TeletexString {
        /// Inner value
        inner: StrOwned,
    }

    impl TeletexString {
        /// Create a new ASN.1 `TeletexString`.
        pub fn new<T>(input: &T) -> Result<Self>
        where
            T: AsRef<[u8]> + ?Sized,
        {
            let input = input.as_ref();

            TeletexStringRef::new(input)?;

            StrOwned::from_bytes(input)
                .map(|inner| Self { inner })
                .map_err(|_| Self::TAG.value_error())
        }
    }

    impl_teletex_string!(TeletexString);

    impl Deref for TeletexString {
        type Target = StrOwned;

        fn deref(&self) -> &Self::Target {
            &self.inner
        }
    }

    impl<'a> From<TeletexStringRef<'a>> for TeletexString {
        fn from(value: TeletexStringRef<'a>) -> TeletexString {
            let inner =
                StrOwned::from_bytes(value.inner.as_bytes()).expect("Invalid TeletexString");
            Self { inner }
        }
    }

    impl<'a> From<&'a TeletexString> for AnyRef<'a> {
        fn from(teletex_string: &'a TeletexString) -> AnyRef<'a> {
            AnyRef::from_tag_and_value(
                Tag::TeletexString,
                BytesRef::new(teletex_string.inner.as_bytes()).expect("Invalid TeletexString"),
            )
        }
    }

    impl<'a> RefToOwned<'a> for TeletexStringRef<'a> {
        type Owned = TeletexString;
        fn ref_to_owned(&self) -> Self::Owned {
            TeletexString {
                inner: self.inner.ref_to_owned(),
            }
        }
    }

    impl OwnedToRef for TeletexString {
        type Borrowed<'a> = TeletexStringRef<'a>;
        fn owned_to_ref(&self) -> Self::Borrowed<'_> {
            TeletexStringRef {
                inner: self.inner.owned_to_ref(),
            }
        }
    }

    impl TryFrom<String> for TeletexString {
        type Error = Error;

        fn try_from(input: String) -> Result<Self> {
            TeletexStringRef::new(&input)?;

            StrOwned::new(input)
                .map(|inner| Self { inner })
                .map_err(|_| Self::TAG.value_error())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TeletexStringRef;
    use crate::Decode;
    use crate::SliceWriter;

    #[test]
    fn parse_bytes() {
        let example_bytes = &[
            0x14, 0x0b, 0x54, 0x65, 0x73, 0x74, 0x20, 0x55, 0x73, 0x65, 0x72, 0x20, 0x31,
        ];

        let teletex_string = TeletexStringRef::from_der(example_bytes).unwrap();
        assert_eq!(teletex_string.as_str(), "Test User 1");
        let mut out = [0_u8; 30];
        let mut writer = SliceWriter::new(&mut out);
        writer.encode(&teletex_string).unwrap();
        let encoded = writer.finish().unwrap();
        assert_eq!(encoded, example_bytes);
    }
}
