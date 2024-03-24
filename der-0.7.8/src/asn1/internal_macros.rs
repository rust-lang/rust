macro_rules! impl_any_conversions {
    ($type: ty) => {
        impl_any_conversions!($type, );
    };
    ($type: ty, $($li: lifetime)?) => {
        impl<'__der: $($li),*, $($li),*> TryFrom<$crate::AnyRef<'__der>> for $type {
            type Error = $crate::Error;

            fn try_from(any: $crate::AnyRef<'__der>) -> Result<$type> {
                any.decode_as()
            }
        }

        #[cfg(feature = "alloc")]
        impl<'__der: $($li),*, $($li),*> TryFrom<&'__der $crate::Any> for $type {
            type Error = $crate::Error;

            fn try_from(any: &'__der $crate::Any) -> Result<$type> {
                any.decode_as()
            }
        }
    };
}

macro_rules! impl_string_type {
    ($type: ty, $($li: lifetime)?) => {
        impl_any_conversions!($type, $($li),*);

        mod __impl_string {
            use super::*;

            use crate::{
                ord::OrdIsValueOrd, BytesRef, DecodeValue, EncodeValue, Header, Length, Reader,
                Result, Writer,
            };
            use core::{fmt, str};

            impl<$($li),*> AsRef<str> for $type {
                fn as_ref(&self) -> &str {
                    self.as_str()
                }
            }

            impl<$($li),*> AsRef<[u8]> for $type {
                fn as_ref(&self) -> &[u8] {
                    self.as_bytes()
                }
            }

            impl<'__der: $($li),*, $($li),*> DecodeValue<'__der> for $type {
                fn decode_value<R: Reader<'__der>>(reader: &mut R, header: Header) -> Result<Self> {
                    Self::new(BytesRef::decode_value(reader, header)?.as_slice())
                }
            }

            impl<$($li),*> EncodeValue for $type {
                fn value_len(&self) -> Result<Length> {
                    self.inner.value_len()
                }

                fn encode_value(&self, writer: &mut impl Writer) -> Result<()> {
                    self.inner.encode_value(writer)
                }
            }

            impl<$($li),*> OrdIsValueOrd for $type {}

            impl<$($li),*> fmt::Display for $type {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str(self.as_str())
                }
            }
        }
    };
}
