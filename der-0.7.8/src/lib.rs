#![no_std]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![doc = include_str!("../README.md")]
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/RustCrypto/media/6ee8e381/logo.svg",
    html_favicon_url = "https://raw.githubusercontent.com/RustCrypto/media/6ee8e381/logo.svg"
)]
#![forbid(unsafe_code)]
#![warn(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::checked_conversions,
    clippy::implicit_saturating_sub,
    clippy::integer_arithmetic,
    clippy::mod_module_files,
    clippy::panic,
    clippy::panic_in_result_fn,
    clippy::unwrap_used,
    missing_docs,
    rust_2018_idioms,
    unused_lifetimes,
    unused_qualifications
)]

//! # Usage
//! ## [`Decode`] and [`Encode`] traits
//! The [`Decode`] and [`Encode`] traits provide the decoding/encoding API
//! respectively, and are designed to work in conjunction with concrete ASN.1
//! types, including all types which impl the [`Sequence`] trait.
//!
//! The traits are impl'd for the following Rust core types:
//! - `()`: ASN.1 `NULL`. See also [`Null`].
//! - [`bool`]: ASN.1 `BOOLEAN`.
//! - [`i8`], [`i16`], [`i32`], [`i64`], [`i128`]: ASN.1 `INTEGER`.
//! - [`u8`], [`u16`], [`u32`], [`u64`], [`u128`]: ASN.1 `INTEGER`.
//! - [`f64`]: ASN.1 `REAL` (gated on `real` crate feature)
//! - [`str`], [`String`][`alloc::string::String`]: ASN.1 `UTF8String`.
//!   `String` requires `alloc` feature. See also [`Utf8StringRef`].
//! - [`Option`]: ASN.1 `OPTIONAL`.
//! - [`SystemTime`][`std::time::SystemTime`]: ASN.1 `GeneralizedTime`. Requires `std` feature.
//! - [`Vec`][`alloc::vec::Vec`]: ASN.1 `SEQUENCE OF`. Requires `alloc` feature.
//! - `[T; N]`: ASN.1 `SEQUENCE OF`. See also [`SequenceOf`].
//!
//! The following ASN.1 types provided by this crate also impl these traits:
//! - [`Any`], [`AnyRef`]: ASN.1 `ANY`.
//! - [`BitString`], [`BitStringRef`]: ASN.1 `BIT STRING`
//! - [`GeneralizedTime`]: ASN.1 `GeneralizedTime`.
//! - [`Ia5StringRef`]: ASN.1 `IA5String`.
//! - [`Null`]: ASN.1 `NULL`.
//! - [`ObjectIdentifier`]: ASN.1 `OBJECT IDENTIFIER`.
//! - [`OctetString`], [`OctetStringRef`]: ASN.1 `OCTET STRING`.
//! - [`PrintableStringRef`]: ASN.1 `PrintableString` (ASCII subset).
//! - [`TeletexStringRef`]: ASN.1 `TeletexString`.
//! - [`VideotexStringRef`]: ASN.1 `VideotexString`.
//! - [`SequenceOf`]: ASN.1 `SEQUENCE OF`.
//! - [`SetOf`], [`SetOfVec`]: ASN.1 `SET OF`.
//! - [`UintRef`]: ASN.1 unsigned `INTEGER` with raw access to encoded bytes.
//! - [`UtcTime`]: ASN.1 `UTCTime`.
//! - [`Utf8StringRef`]: ASN.1 `UTF8String`.
//!
//! Context specific fields can be modeled using these generic types:
//! - [`ContextSpecific`]: decoder/encoder for owned context-specific fields
//! - [`ContextSpecificRef`]: encode-only type for references to context-specific fields
//!
//! ## Example
//! The following example implements X.509's `AlgorithmIdentifier` message type
//! as defined in [RFC 5280 Section 4.1.1.2].
//!
//! The ASN.1 schema for this message type is as follows:
//!
//! ```text
//! AlgorithmIdentifier  ::=  SEQUENCE  {
//!      algorithm               OBJECT IDENTIFIER,
//!      parameters              ANY DEFINED BY algorithm OPTIONAL  }
//! ```
//!
//! Structured ASN.1 messages are typically encoded as a `SEQUENCE`, which
//! this crate maps to a Rust struct using the [`Sequence`] trait. This
//! trait is bounded on the [`Decode`] trait and provides a blanket impl
//! of the [`Encode`] trait, so any type which impls [`Sequence`] can be
//! used for both decoding and encoding.
//!
//! The following code example shows how to define a struct which maps to the
//! above schema, as well as impl the [`Sequence`] trait for that struct:
//!
//! ```
//! # #[cfg(all(feature = "alloc", feature = "oid"))]
//! # {
//! // Note: the following example does not require the `std` feature at all.
//! // It does leverage the `alloc` feature, but also provides instructions for
//! // "heapless" usage when the `alloc` feature is disabled.
//! use der::{
//!     asn1::{AnyRef, ObjectIdentifier},
//!     DecodeValue, Decode, SliceReader, Encode, Header, Reader, Sequence
//! };
//!
//! /// X.509 `AlgorithmIdentifier`.
//! #[derive(Copy, Clone, Debug, Eq, PartialEq)]
//! pub struct AlgorithmIdentifier<'a> {
//!     /// This field contains an ASN.1 `OBJECT IDENTIFIER`, a.k.a. OID.
//!     pub algorithm: ObjectIdentifier,
//!
//!     /// This field is `OPTIONAL` and contains the ASN.1 `ANY` type, which
//!     /// in this example allows arbitrary algorithm-defined parameters.
//!     pub parameters: Option<AnyRef<'a>>
//! }
//!
//! impl<'a> DecodeValue<'a> for AlgorithmIdentifier<'a> {
//!     fn decode_value<R: Reader<'a>>(reader: &mut R, _header: Header) -> der::Result<Self> {
//!        // The `der::Decoder::Decode` method can be used to decode any
//!        // type which impls the `Decode` trait, which is impl'd for
//!        // all of the ASN.1 built-in types in the `der` crate.
//!        //
//!        // Note that if your struct's fields don't contain an ASN.1
//!        // built-in type specifically, there are also helper methods
//!        // for all of the built-in types supported by this library
//!        // which can be used to select a specific type.
//!        //
//!        // For example, another way of decoding this particular field,
//!        // which contains an ASN.1 `OBJECT IDENTIFIER`, is by calling
//!        // `decoder.oid()`. Similar methods are defined for other
//!        // ASN.1 built-in types.
//!        let algorithm = reader.decode()?;
//!
//!        // This field contains an ASN.1 `OPTIONAL` type. The `der` crate
//!        // maps this directly to Rust's `Option` type and provides
//!        // impls of the `Decode` and `Encode` traits for `Option`.
//!        // To explicitly request an `OPTIONAL` type be decoded, use the
//!        // `decoder.optional()` method.
//!        let parameters = reader.decode()?;
//!
//!        // The value returned from the provided `FnOnce` will be
//!        // returned from the `any.sequence(...)` call above.
//!        // Note that the entire sequence body *MUST* be consumed
//!        // or an error will be returned.
//!        Ok(Self { algorithm, parameters })
//!     }
//! }
//!
//! impl<'a> ::der::EncodeValue for AlgorithmIdentifier<'a> {
//!     fn value_len(&self) -> ::der::Result<::der::Length> {
//!         self.algorithm.encoded_len()? + self.parameters.encoded_len()?
//!     }
//!
//!     fn encode_value(&self, writer: &mut impl ::der::Writer) -> ::der::Result<()> {
//!         self.algorithm.encode(writer)?;
//!         self.parameters.encode(writer)?;
//!         Ok(())
//!     }
//! }
//!
//! impl<'a> Sequence<'a> for AlgorithmIdentifier<'a> {}
//!
//! // Example parameters value: OID for the NIST P-256 elliptic curve.
//! let parameters = "1.2.840.10045.3.1.7".parse::<ObjectIdentifier>().unwrap();
//!
//! // We need to convert `parameters` into an `Any<'a>` type, which wraps a
//! // `&'a [u8]` byte slice.
//! //
//! // To do that, we need owned DER-encoded data so that we can have
//! // `AnyRef` borrow a reference to it, so we have to serialize the OID.
//! //
//! // When the `alloc` feature of this crate is enabled, any type that impls
//! // the `Encode` trait including all ASN.1 built-in types and any type
//! // which impls `Sequence` can be serialized by calling `Encode::to_der()`.
//! //
//! // If you would prefer to avoid allocations, you can create a byte array
//! // as backing storage instead, pass that to `der::Encoder::new`, and then
//! // encode the `parameters` value using `encoder.encode(parameters)`.
//! let der_encoded_parameters = parameters.to_der().unwrap();
//!
//! let algorithm_identifier = AlgorithmIdentifier {
//!     // OID for `id-ecPublicKey`, if you're curious
//!     algorithm: "1.2.840.10045.2.1".parse().unwrap(),
//!
//!     // `Any<'a>` impls `TryFrom<&'a [u8]>`, which parses the provided
//!     // slice as an ASN.1 DER-encoded message.
//!     parameters: Some(der_encoded_parameters.as_slice().try_into().unwrap())
//! };
//!
//! // Serialize the `AlgorithmIdentifier` created above as ASN.1 DER,
//! // allocating a `Vec<u8>` for storage.
//! //
//! // As mentioned earlier, if you don't have the `alloc` feature enabled you
//! // can create a fix-sized array instead, then call `Encoder::new` with a
//! // reference to it, then encode the message using
//! // `encoder.encode(algorithm_identifier)`, then finally `encoder.finish()`
//! // to obtain a byte slice containing the encoded message.
//! let der_encoded_algorithm_identifier = algorithm_identifier.to_der().unwrap();
//!
//! // Deserialize the `AlgorithmIdentifier` we just serialized from ASN.1 DER
//! // using `der::Decode::from_bytes`.
//! let decoded_algorithm_identifier = AlgorithmIdentifier::from_der(
//!     &der_encoded_algorithm_identifier
//! ).unwrap();
//!
//! // Ensure the original `AlgorithmIdentifier` is the same as the one we just
//! // decoded from ASN.1 DER.
//! assert_eq!(algorithm_identifier, decoded_algorithm_identifier);
//! # }
//! ```
//!
//! ## Custom derive support
//! When the `derive` feature of this crate is enabled, the following custom
//! derive macros are available:
//!
//! - [`Choice`]: derive for `CHOICE` enum (see [`der_derive::Choice`])
//! - [`Enumerated`]: derive for `ENUMERATED` enum (see [`der_derive::Enumerated`])
//! - [`Sequence`]: derive for `SEQUENCE` struct (see [`der_derive::Sequence`])
//!
//! ### Derive [`Sequence`] for struct
//! The following is a code example of how to use the [`Sequence`] custom derive:
//!
//! ```
//! # #[cfg(all(feature = "alloc", feature = "derive", feature = "oid"))]
//! # {
//! use der::{asn1::{AnyRef, ObjectIdentifier}, Encode, Decode, Sequence};
//!
//! /// X.509 `AlgorithmIdentifier` (same as above)
//! #[derive(Copy, Clone, Debug, Eq, PartialEq, Sequence)] // NOTE: added `Sequence`
//! pub struct AlgorithmIdentifier<'a> {
//!     /// This field contains an ASN.1 `OBJECT IDENTIFIER`, a.k.a. OID.
//!     pub algorithm: ObjectIdentifier,
//!
//!     /// This field is `OPTIONAL` and contains the ASN.1 `ANY` type, which
//!     /// in this example allows arbitrary algorithm-defined parameters.
//!     pub parameters: Option<AnyRef<'a>>
//! }
//!
//! // Example parameters value: OID for the NIST P-256 elliptic curve.
//! let parameters_oid = "1.2.840.10045.3.1.7".parse::<ObjectIdentifier>().unwrap();
//!
//! let algorithm_identifier = AlgorithmIdentifier {
//!     // OID for `id-ecPublicKey`, if you're curious
//!     algorithm: "1.2.840.10045.2.1".parse().unwrap(),
//!
//!     // `Any<'a>` impls `From<&'a ObjectIdentifier>`, allowing OID constants to
//!     // be directly converted to an `AnyRef` type for this use case.
//!     parameters: Some(AnyRef::from(&parameters_oid))
//! };
//!
//! // Encode
//! let der_encoded_algorithm_identifier = algorithm_identifier.to_der().unwrap();
//!
//! // Decode
//! let decoded_algorithm_identifier = AlgorithmIdentifier::from_der(
//!     &der_encoded_algorithm_identifier
//! ).unwrap();
//!
//! assert_eq!(algorithm_identifier, decoded_algorithm_identifier);
//! # }
//! ```
//!
//! For fields which don't directly impl [`Decode`] and [`Encode`],
//! you can add annotations to convert to an intermediate ASN.1 type
//! first, so long as that type impls `TryFrom` and `Into` for the
//! ASN.1 type.
//!
//! For example, structs containing `&'a [u8]` fields may want them encoded
//! as either a `BIT STRING` or `OCTET STRING`. By using the
//! `#[asn1(type = "BIT STRING")]` annotation it's possible to select which
//! ASN.1 type should be used.
//!
//! Building off the above example:
//!
//! ```rust
//! # #[cfg(all(feature = "alloc", feature = "derive", feature = "oid"))]
//! # {
//! # use der::{asn1::{AnyRef, BitStringRef, ObjectIdentifier}, Sequence};
//! #
//! # #[derive(Copy, Clone, Debug, Eq, PartialEq, Sequence)]
//! # pub struct AlgorithmIdentifier<'a> {
//! #     pub algorithm: ObjectIdentifier,
//! #     pub parameters: Option<AnyRef<'a>>
//! # }
//! /// X.509 `SubjectPublicKeyInfo` (SPKI)
//! #[derive(Copy, Clone, Debug, Eq, PartialEq, Sequence)]
//! pub struct SubjectPublicKeyInfo<'a> {
//!     /// X.509 `AlgorithmIdentifier`
//!     pub algorithm: AlgorithmIdentifier<'a>,
//!
//!     /// Public key data
//!     pub subject_public_key: BitStringRef<'a>,
//! }
//! # }
//! ```
//!
//! # See also
//! For more information about ASN.1 DER we recommend the following guides:
//!
//! - [A Layman's Guide to a Subset of ASN.1, BER, and DER] (RSA Laboratories)
//! - [A Warm Welcome to ASN.1 and DER] (Let's Encrypt)
//!
//! [RFC 5280 Section 4.1.1.2]: https://tools.ietf.org/html/rfc5280#section-4.1.1.2
//! [A Layman's Guide to a Subset of ASN.1, BER, and DER]: https://luca.ntop.org/Teaching/Appunti/asn1.html
//! [A Warm Welcome to ASN.1 and DER]: https://letsencrypt.org/docs/a-warm-welcome-to-asn1-and-der/
//!
//! [`Any`]: asn1::Any
//! [`AnyRef`]: asn1::AnyRef
//! [`ContextSpecific`]: asn1::ContextSpecific
//! [`ContextSpecificRef`]: asn1::ContextSpecificRef
//! [`BitString`]: asn1::BitString
//! [`BitStringRef`]: asn1::BitStringRef
//! [`GeneralizedTime`]: asn1::GeneralizedTime
//! [`Ia5StringRef`]: asn1::Ia5StringRef
//! [`Null`]: asn1::Null
//! [`ObjectIdentifier`]: asn1::ObjectIdentifier
//! [`OctetString`]: asn1::OctetString
//! [`OctetStringRef`]: asn1::OctetStringRef
//! [`PrintableStringRef`]: asn1::PrintableStringRef
//! [`TeletexStringRef`]: asn1::TeletexStringRef
//! [`VideotexStringRef`]: asn1::VideotexStringRef
//! [`SequenceOf`]: asn1::SequenceOf
//! [`SetOf`]: asn1::SetOf
//! [`SetOfVec`]: asn1::SetOfVec
//! [`UintRef`]: asn1::UintRef
//! [`UtcTime`]: asn1::UtcTime
//! [`Utf8StringRef`]: asn1::Utf8StringRef

#[cfg(feature = "alloc")]
#[allow(unused_imports)]
#[macro_use]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod asn1;
pub mod referenced;

pub(crate) mod arrayvec;
mod bytes_ref;
mod datetime;
mod decode;
mod encode;
mod encode_ref;
mod error;
mod header;
mod length;
mod ord;
mod reader;
mod str_ref;
mod tag;
mod writer;

#[cfg(feature = "alloc")]
mod bytes_owned;
#[cfg(feature = "alloc")]
mod document;
#[cfg(feature = "alloc")]
mod str_owned;

pub use crate::{
    asn1::{AnyRef, Choice, Sequence},
    datetime::DateTime,
    decode::{Decode, DecodeOwned, DecodeValue},
    encode::{Encode, EncodeValue},
    encode_ref::{EncodeRef, EncodeValueRef},
    error::{Error, ErrorKind, Result},
    header::Header,
    length::{IndefiniteLength, Length},
    ord::{DerOrd, ValueOrd},
    reader::{nested::NestedReader, slice::SliceReader, Reader},
    tag::{Class, FixedTag, Tag, TagMode, TagNumber, Tagged},
    writer::{slice::SliceWriter, Writer},
};

#[cfg(feature = "alloc")]
pub use crate::{asn1::Any, document::Document};

#[cfg(feature = "bigint")]
pub use crypto_bigint as bigint;

#[cfg(feature = "derive")]
pub use der_derive::{Choice, Enumerated, Sequence, ValueOrd};

#[cfg(feature = "flagset")]
pub use flagset;

#[cfg(feature = "oid")]
pub use const_oid as oid;

#[cfg(feature = "pem")]
pub use {
    crate::{decode::DecodePem, encode::EncodePem, reader::pem::PemReader, writer::pem::PemWriter},
    pem_rfc7468 as pem,
};

#[cfg(feature = "time")]
pub use time;

#[cfg(feature = "zeroize")]
pub use zeroize;

#[cfg(all(feature = "alloc", feature = "zeroize"))]
pub use crate::document::SecretDocument;

pub(crate) use crate::{arrayvec::ArrayVec, bytes_ref::BytesRef, str_ref::StrRef};
#[cfg(feature = "alloc")]
pub(crate) use crate::{bytes_owned::BytesOwned, str_owned::StrOwned};
