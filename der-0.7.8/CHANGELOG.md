# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.7.8 (2023-08-07)
### Added
- `bytes` feature ([#1156])
- impl `RefToOwned`/`OwnedToRef` for `&[u8]`/`Box<[u8]>` ([#1188])
- `BmpString` ([#1164])

### Changed
- no-panic cleanup ([#1169])
- Bump `der_derive` dependency to v0.7.2 ([#1192])

[#1156]: https://github.com/RustCrypto/formats/pull/1156
[#1164]: https://github.com/RustCrypto/formats/pull/1164
[#1169]: https://github.com/RustCrypto/formats/pull/1169
[#1188]: https://github.com/RustCrypto/formats/pull/1188
[#1192]: https://github.com/RustCrypto/formats/pull/1192

## 0.7.7 (2023-06-29)
### Added
- `TryFrom<String>` impl for strings based on `StrOwned` ([#1064])

[#1064]: https://github.com/RustCrypto/formats/pull/1064

## 0.7.6 (2023-05-16)
### Added
- `SetOfVec::{extend, from_iter}` methods ([#1065])
- `SetOf(Vec)::{insert, insert_ordered}` methods ([#1067])

### Changed
- Deprecate `SetOf(Vec)::add` ([#1067])

### Fixed
- Off-by-one error in `BMPString` tag ([#1037])
- Handling of non-unique items in `SetOf`(Vec) ([#1066])

[#1037]: https://github.com/RustCrypto/formats/pull/1037
[#1065]: https://github.com/RustCrypto/formats/pull/1065
[#1066]: https://github.com/RustCrypto/formats/pull/1066
[#1067]: https://github.com/RustCrypto/formats/pull/1067

## 0.7.5 (2023-04-24)
### Added
- adds support for `DateTime::INFINITY` ([#1026])

[#1026]: https://github.com/RustCrypto/formats/pull/1026

## 0.7.4 (2023-04-19)
### Added
- `Decode` and `Encode` impls for `PhantomData` ([#1009])
- `ValueOrd` and `DerOrd` impls for `PhantomData` ([#1012])

### Changed
- Bump `hex-literal` dependency to v0.4.1 ([#999])
- Bump `der_derive` dependency to v0.7.1 ([#1016])

[#1009]: https://github.com/RustCrypto/formats/pull/1009
[#1012]: https://github.com/RustCrypto/formats/pull/1012
[#1016]: https://github.com/RustCrypto/formats/pull/1016

## 0.7.3 (2023-04-06)
### Added
- `UtcTime::MAX_YEAR` associated constant ([#989])

[#989]: https://github.com/RustCrypto/formats/pull/989

## 0.7.2 (2023-04-04)
### Added
- Expose `NestedReader ([#925])
- `From<ObjectIdentifier>` impl for `Any` ([#965])
- `Any::null` helper ([#969])
- `Any::encode_from` ([#976])

[#925]: https://github.com/RustCrypto/formats/pull/925
[#965]: https://github.com/RustCrypto/formats/pull/965
[#969]: https://github.com/RustCrypto/formats/pull/969
[#976]: https://github.com/RustCrypto/formats/pull/976

## 0.7.1 (2023-03-07)
### Changed
- Make `zeroize`'s `alloc` feature conditional ([#920])

[#920]: https://github.com/RustCrypto/formats/pull/920

## 0.7.0 (2023-02-26) [YANKED]
### Added
- `OwnedtoRef`/`RefToOwned` traits; MSRV 1.65 ([#797])
- `OctetStringRef::decode_into` ([#817])
- `Int` and `IntRef` types ([#823])
- `IndefiniteLength` type ([#830])
- `Any::value` accessor ([#833])
- Buffered PEM reader ([#839])
- `OctetString::into_bytes` ([#845])
- Blanket impls on `Box<T>` for `DecodeValue`, `EncodeValue`, and `Sequence` ([#860])

### Changed
- Rename `UIntRef` => `UintRef` ([#786])
- Replace use of `dyn Writer` with `impl Writer` ([#828])
- Rename `AnyRef::decode_into` -> `::decode_as` ([#829])
- Bump `pem-rfc7468` dependency to v0.7 ([#894])
- Rename `Encode::to_vec` => `::to_der` ([#898])

### Removed
- `Sequence::fields` method ([#828])
- Inherent `AnyRef` decoding methods ([#829])

[#786]: https://github.com/RustCrypto/formats/pull/786
[#797]: https://github.com/RustCrypto/formats/pull/797
[#817]: https://github.com/RustCrypto/formats/pull/817
[#823]: https://github.com/RustCrypto/formats/pull/823
[#828]: https://github.com/RustCrypto/formats/pull/828
[#829]: https://github.com/RustCrypto/formats/pull/829
[#830]: https://github.com/RustCrypto/formats/pull/830
[#833]: https://github.com/RustCrypto/formats/pull/833
[#839]: https://github.com/RustCrypto/formats/pull/839
[#845]: https://github.com/RustCrypto/formats/pull/845
[#860]: https://github.com/RustCrypto/formats/pull/860
[#894]: https://github.com/RustCrypto/formats/pull/894
[#898]: https://github.com/RustCrypto/formats/pull/898

## 0.6.1 (2022-12-05)
### Added
- Rudimentary implementation of `TeletexString` and `VideotexString` ([#691])
- Impl `ValueOrd` for `FlagSet<T>` and `UIntRef` ([#723])

### Changed
- Eliminate some boilerplate code by using `Deref` ([#697])

[#691]: https://github.com/RustCrypto/formats/pull/691
[#697]: https://github.com/RustCrypto/formats/pull/697
[#723]: https://github.com/RustCrypto/formats/pull/723

## 0.6.0 (2022-05-08)
### Added
- Impl `ValueOrd` for `SetOf` and `SetOfVec` ([#362])
- `SequenceRef` type ([#374])
- Support for `SetOf` sorting on heapless `no_std` targets ([#401])
- Support for mapping `BitString` to/from a `FlagSet` ([#412])
- `DecodeOwned` marker trait ([#529])
- Support for the ASN.1 `REAL` type ([#346])
- `DecodePem` and `EncodePem` traits ([#571])
- `Document` and `SecretDocument` types ([#571])
- `EncodeRef`/`EncodeValueRef` wrapper types ([#604])
- `Writer` trait ([#605])
- `Reader` trait ([#606])
- Streaming on-the-fly `PemReader` and `PemWriter` ([#618], [#636])
- Owned `BitString` ([#636])
- Owned `Any` and `OctetString` types ([#640])

### Changed
- Pass `Header` to `DecodeValue` ([#392])
- Bump `const-oid` dependency to v0.9 ([#507])
- Renamed `Decodable`/`Encodable` => `Decode`/`Encode` ([#523])
- Enable arithmetic, casting, and panic `clippy` lints ([#556], [#579])
- Use `&mut dyn Writer` as output for `Encode::encode` and `EncodeValue::encode_value` ([#611])
- Bump `pem-rfc7468` dependency to v0.6 ([#620])
- Use `Reader<'a>` as input for `Decode::decode` and `DecodeValue::decode_value` ([#633])
- Renamed `Any` => `AnyRef` ([#637])
- Renamed `BitString` => `BitStringRef` ([#637])
- Renamed `Ia5String` => `Ia5StringRef` ([#637])
- Renamed `OctetString` => `OctetStringRef` ([#637])
- Renamed `PrintableString` => `PrintableStringRef` ([#637])
- Renamed `Utf8String` => `Utf8StringRef` ([#637])
- Renamed `UIntBytes` => `UIntRef` ([#637])
- Renamed `Decoder` => `SliceReader` ([#651])
- Renamed `Encoder` => `SliceWriter` ([#651])

### Fixed
- Handling of oversized unsigned `INTEGER` inputs ([#447])

### Removed
- `bigint` feature ([#344])
- `OrdIsValueOrd` trait ([#359])
- `Document` trait ([#571])
- `OptionalRef` ([#604])
- Decode-time SET OF ordering checks ([#625])

[#344]: https://github.com/RustCrypto/formats/pull/344
[#346]: https://github.com/RustCrypto/formats/pull/346
[#359]: https://github.com/RustCrypto/formats/pull/359
[#362]: https://github.com/RustCrypto/formats/pull/362
[#374]: https://github.com/RustCrypto/formats/pull/374
[#392]: https://github.com/RustCrypto/formats/pull/392
[#401]: https://github.com/RustCrypto/formats/pull/401
[#412]: https://github.com/RustCrypto/formats/pull/412
[#447]: https://github.com/RustCrypto/formats/pull/447
[#507]: https://github.com/RustCrypto/formats/pull/507
[#523]: https://github.com/RustCrypto/formats/pull/523
[#529]: https://github.com/RustCrypto/formats/pull/529
[#556]: https://github.com/RustCrypto/formats/pull/556
[#571]: https://github.com/RustCrypto/formats/pull/571
[#579]: https://github.com/RustCrypto/formats/pull/579
[#604]: https://github.com/RustCrypto/formats/pull/604
[#605]: https://github.com/RustCrypto/formats/pull/605
[#606]: https://github.com/RustCrypto/formats/pull/606
[#611]: https://github.com/RustCrypto/formats/pull/611
[#618]: https://github.com/RustCrypto/formats/pull/618
[#620]: https://github.com/RustCrypto/formats/pull/620
[#625]: https://github.com/RustCrypto/formats/pull/625
[#633]: https://github.com/RustCrypto/formats/pull/633
[#636]: https://github.com/RustCrypto/formats/pull/636
[#637]: https://github.com/RustCrypto/formats/pull/637
[#640]: https://github.com/RustCrypto/formats/pull/640
[#651]: https://github.com/RustCrypto/formats/pull/651

## 0.5.1 (2021-11-17)
### Added
- `Any::NULL` constant ([#226])

[#226]: https://github.com/RustCrypto/formats/pull/226

## 0.5.0 (2021-11-15) [YANKED]
### Added
- Support for `IMPLICIT` mode `CONTEXT-SPECIFIC` fields ([#61])
- `DecodeValue`/`EncodeValue` traits ([#63])
- Expose `DateTime` through public API ([#75])
- `SEQUENCE OF` support for `[T; N]` ([#90])
- `SequenceOf` type ([#95])
- `SEQUENCE OF` support for `Vec` ([#96])
- `Document` trait ([#117])
- Basic integration with `time` crate ([#129])
- `Tag::NumericString` ([#132])
- Support for unused bits to `BitString` ([#141])
- `Decoder::{peek_tag, peek_header}` ([#142])
- Type hint in `encoder `sequence` method ([#147])
- `Tag::Enumerated` ([#153])
- `ErrorKind::TagNumberInvalid` ([#156])
- `Tag::VisibleString` and `Tag::BmpString` ([#160])
- Inherent constants for all valid `TagNumber`s ([#165])
- `DerOrd` and `ValueOrd` traits ([#190])
- `ContextSpecificRef` type ([#199])

### Changed
- Make `ContextSpecific` generic around an inner type ([#60])
- Removed `SetOf` trait; rename `SetOfArray` => `SetOf` ([#97])
- Rename `Message` trait to `Sequence` ([#99])
- Make `GeneralizedTime`/`UtcTime` into `DateTime` newtypes ([#102])
- Rust 2021 edition upgrade; MSRV 1.56 ([#136])
- Replace `ErrorKind::Truncated` with `ErrorKind::Incomplete` ([#143])
- Rename `ErrorKind::UnknownTagMode` => `ErrorKind::TagModeUnknown` ([#155])
- Rename `ErrorKind::UnexpectedTag` => `ErrorKind::TagUnexpected` ([#155])
- Rename `ErrorKind::UnknownTag` => `ErrorKind::TagUnknown` ([#155])
- Consolidate `ErrorKind::{Incomplete, Underlength}` ([#157])
- Rename `Tagged` => `FixedTag`; add new `Tagged` trait ([#189])
- Use `DerOrd` for `SetOf*` types ([#200])
- Switch `impl From<BitString> for &[u8]` to `TryFrom` ([#203])
- Bump `crypto-bigint` dependency to v0.3 ([#215])
- Bump `const-oid` dependency to v0.7 ([#216])
- Bump `pem-rfc7468` dependency to v0.3 ([#217])
- Bump `der_derive` dependency to v0.5 ([#221])

### Removed
- `Sequence` struct ([#98])
- `Tagged` bound on `ContextSpecific::decode_implicit` ([#161])
- `ErrorKind::DuplicateField` ([#162])

[#60]: https://github.com/RustCrypto/formats/pull/60
[#61]: https://github.com/RustCrypto/formats/pull/61
[#63]: https://github.com/RustCrypto/formats/pull/63
[#75]: https://github.com/RustCrypto/formats/pull/75
[#90]: https://github.com/RustCrypto/formats/pull/90
[#95]: https://github.com/RustCrypto/formats/pull/95
[#96]: https://github.com/RustCrypto/formats/pull/96
[#97]: https://github.com/RustCrypto/formats/pull/97
[#98]: https://github.com/RustCrypto/formats/pull/98
[#99]: https://github.com/RustCrypto/formats/pull/99
[#102]: https://github.com/RustCrypto/formats/pull/102
[#117]: https://github.com/RustCrypto/formats/pull/117
[#129]: https://github.com/RustCrypto/formats/pull/129
[#132]: https://github.com/RustCrypto/formats/pull/132
[#136]: https://github.com/RustCrypto/formats/pull/136
[#141]: https://github.com/RustCrypto/formats/pull/141
[#142]: https://github.com/RustCrypto/formats/pull/142
[#143]: https://github.com/RustCrypto/formats/pull/143
[#147]: https://github.com/RustCrypto/formats/pull/147
[#153]: https://github.com/RustCrypto/formats/pull/153
[#155]: https://github.com/RustCrypto/formats/pull/155
[#156]: https://github.com/RustCrypto/formats/pull/156
[#157]: https://github.com/RustCrypto/formats/pull/157
[#160]: https://github.com/RustCrypto/formats/pull/160
[#161]: https://github.com/RustCrypto/formats/pull/161
[#162]: https://github.com/RustCrypto/formats/pull/162
[#165]: https://github.com/RustCrypto/formats/pull/165
[#189]: https://github.com/RustCrypto/formats/pull/189
[#190]: https://github.com/RustCrypto/formats/pull/190
[#199]: https://github.com/RustCrypto/formats/pull/199
[#200]: https://github.com/RustCrypto/formats/pull/200
[#203]: https://github.com/RustCrypto/formats/pull/203
[#215]: https://github.com/RustCrypto/formats/pull/215
[#216]: https://github.com/RustCrypto/formats/pull/216
[#217]: https://github.com/RustCrypto/formats/pull/217
[#221]: https://github.com/RustCrypto/formats/pull/221

## 0.4.5 (2021-12-01)
### Fixed
- Backport [#147] type hint fix for WASM platforms to 0.4.x

## 0.4.4 (2021-10-06)
### Removed
- Accidentally checked-in `target/` directory ([#66])

[#66]: https://github.com/RustCrypto/formats/pull/66

## 0.4.3 (2021-09-15)
### Added
- `Tag::unexpected_error` ([#33])

[#33]: https://github.com/RustCrypto/formats/pull/33

## 0.4.2 (2021-09-14)
### Changed
- Moved to `formats` repo ([#2])

### Fixed
- ASN.1 `SET` type now flagged with the constructed bit

[#2]: https://github.com/RustCrypto/formats/pull/2

## 0.4.1 (2021-08-08)
### Fixed
- Encoding `UTCTime` for dates with `20xx` years

## 0.4.0 (2021-06-07)
### Added
- `TagNumber` type
- Const generic integer de/encoders with support for all of Rust's integer
  primitives
- `crypto-bigint` support
- `Tag` number helpers
- `Tag::octet`
- `ErrorKind::Value` helpers
- `SequenceIter`

### Changed
- Bump `const-oid` crate dependency to v0.6
- Make `Tag` structured
- Namespace ASN.1 types in `asn1` module
- Refactor context-specific field decoding
- MSRV 1.51
- Rename `big-uint` crate feature to `bigint`
- Rename `BigUInt` to `UIntBytes`
- Have `Decoder::error()` return an `Error`
  
### Removed
- Deprecated methods replaced by associated constants

## 0.3.5 (2021-05-24)
### Added
- Helper methods for context-specific fields
- `ContextSpecific` field wrapper
- Decoder position tracking for errors during `Any<'a>` decoding

### Fixed
- `From` conversion for `BitString` into `Any`

## 0.3.4 (2021-05-16)
### Changed
- Support `Length` of up to 1 MiB

## 0.3.3 (2021-04-15)
### Added
- `Length` constants

### Changed
- Deprecate `const fn` methods replaced by `Length` constants

## 0.3.2 (2021-04-15)
### Fixed
- Non-critical bug allowing `Length` to exceed the max invariant

## 0.3.1 (2021-04-01) [YANKED]
### Added
- `PartialOrd` + `Ord` impls to all ASN.1 types

## 0.3.0 (2021-03-22) [YANKED]
### Added
- Impl `Decode`/`Encoded`/`Tagged` for `String`
- `Length::one` and `Length::for_tlv`
- `SET OF` support with `SetOf` trait and `SetOfRef`

### Changed
- Rename `Decodable::from_bytes` => `Decodable::from_der`
- Separate `sequence` and `message`
- Rename `ErrorKind::Oid` => `ErrorKind::MalformedOid`
- Auto-derive `From` impls for variants when deriving `Choice`
- Make `Length` use `u32` internally
- Make `Sequence` constructor private
- Bump `const_oid` to v0.5
- Bump `der_derive` to v0.3

### Removed
- Deprecated methods
- `BigUIntSize`

## 0.2.10 (2021-02-28)
### Added
- Impl `From<ObjectIdentifier>` for `Any`

### Changed
- Bump minimum `const-oid` dependency to v0.4.4

## 0.2.9 (2021-02-24)
### Added
- Support for `IA5String`

## 0.2.8 (2021-02-22)
### Added
- `Choice` trait

## 0.2.7 (2021-02-20)
### Added
- Export `Header` publicly
- Make `Encoder::reserve` public

## 0.2.6 (2021-02-19)
### Added
- Make the unit type an encoding of `NULL`

## 0.2.5 (2021-02-18)
### Added
- `ErrorKind::UnknownOid` variant

## 0.2.4 (2021-02-16)
### Added
- `Any::is_null` method

### Changed
- Deprecate `Any::null` method

## 0.2.3 (2021-02-15)
### Added
- Additional `rustdoc` documentation

## 0.2.2 (2021-02-12)
### Added
- Support for `UTCTime` and `GeneralizedTime`

## 0.2.1 (2021-02-02)
### Added
- Support for `PrintableString` and `Utf8String`

## 0.2.0 (2021-01-22)
### Added
- `BigUInt` type
- `i16` support
- `u8` and `u16` support
- Integer decoder helper methods

### Fixed
- Handle leading byte of `BIT STRING`s

## 0.1.0 (2020-12-21)
- Initial release
