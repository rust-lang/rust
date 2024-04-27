// @generated
/// Implement `DataProvider<CollationFallbackSupplementV1Marker>` on the given struct using the data
/// hardcoded in this file. This allows the struct to be used with
/// `icu`'s `_unstable` constructors.
#[doc(hidden)]
#[macro_export]
macro_rules! __impl_fallback_supplement_co_v1 {
    ($ provider : ty) => {
        #[clippy::msrv = "1.66"]
        const _: () = <$provider>::MUST_USE_MAKE_PROVIDER_MACRO;
        #[clippy::msrv = "1.66"]
        impl $provider {
            #[doc(hidden)]
            pub const SINGLETON_FALLBACK_SUPPLEMENT_CO_V1: &'static <icu_locid_transform::provider::CollationFallbackSupplementV1Marker as icu_provider::DataMarker>::Yokeable = &icu_locid_transform::provider::LocaleFallbackSupplementV1 {
                parents: unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x01\0\0\0\0\0yue") }, unsafe { zerovec::ZeroVec::from_bytes_unchecked(b"zh\0\x01Hant\0\0\0\0") })
                },
                unicode_extension_defaults: unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap2d::from_parts_unchecked(unsafe { zerovec::ZeroVec::from_bytes_unchecked(b"co") }, unsafe { zerovec::ZeroVec::from_bytes_unchecked(b"\x02\0\0\0") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x02\0zhzh-Hant") }, unsafe { zerovec::VarZeroVec::from_bytes_unchecked(b"\x02\0\0\0\0\0\x06\0pinyinstroke") })
                },
            };
        }
        #[clippy::msrv = "1.66"]
        impl icu_provider::DataProvider<icu_locid_transform::provider::CollationFallbackSupplementV1Marker> for $provider {
            fn load(&self, req: icu_provider::DataRequest) -> Result<icu_provider::DataResponse<icu_locid_transform::provider::CollationFallbackSupplementV1Marker>, icu_provider::DataError> {
                if req.locale.is_empty() { Ok(icu_provider::DataResponse { payload: Some(icu_provider::DataPayload::from_static_ref(Self::SINGLETON_FALLBACK_SUPPLEMENT_CO_V1)), metadata: Default::default() }) } else { Err(icu_provider::DataErrorKind::ExtraneousLocale.with_req(<icu_locid_transform::provider::CollationFallbackSupplementV1Marker as icu_provider::KeyedDataMarker>::KEY, req)) }
            }
        }
    };
}
