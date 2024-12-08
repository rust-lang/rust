// @generated
/// Implement `DataProvider<LocaleFallbackLikelySubtagsV1Marker>` on the given struct using the data
/// hardcoded in this file. This allows the struct to be used with
/// `icu`'s `_unstable` constructors.
#[doc(hidden)]
#[macro_export]
macro_rules! __impl_fallback_likelysubtags_v1 {
    ($ provider : ty) => {
        #[clippy::msrv = "1.66"]
        const _: () = <$provider>::MUST_USE_MAKE_PROVIDER_MACRO;
        #[clippy::msrv = "1.66"]
        impl $provider {
            #[doc(hidden)]
            pub const SINGLETON_FALLBACK_LIKELYSUBTAGS_V1: &'static <icu_locid_transform::provider::LocaleFallbackLikelySubtagsV1Marker as icu_provider::DataMarker>::Yokeable = &icu_locid_transform::provider::LocaleFallbackLikelySubtagsV1 {
                l2s: unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::ZeroVec::from_bytes_unchecked(b"am\0ar\0as\0be\0bg\0bgcbhobn\0brxchrcv\0doiel\0fa\0gu\0he\0hi\0hy\0ja\0ka\0kk\0km\0kn\0ko\0kokks\0ky\0lo\0maimk\0ml\0mn\0mnimr\0my\0ne\0or\0pa\0ps\0rajru\0sa\0satsd\0si\0sr\0ta\0te\0tg\0th\0ti\0tt\0uk\0ur\0yuezh\0") }, unsafe { zerovec::ZeroVec::from_bytes_unchecked(b"EthiArabBengCyrlCyrlDevaDevaBengDevaCherCyrlDevaGrekArabGujrHebrDevaArmnJpanGeorCyrlKhmrKndaKoreDevaArabCyrlLaooDevaCyrlMlymCyrlBengDevaMymrDevaOryaGuruArabDevaCyrlDevaOlckArabSinhCyrlTamlTeluCyrlThaiEthiCyrlCyrlArabHantHans") })
                },
                lr2s: unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap2d::from_parts_unchecked(unsafe { zerovec::ZeroVec::from_bytes_unchecked(b"az\0ha\0kk\0ky\0mn\0ms\0pa\0sd\0sr\0tg\0uz\0yuezh\0") }, unsafe { zerovec::ZeroVec::from_bytes_unchecked(b"\x03\0\0\0\x05\0\0\0\t\0\0\0\x0B\0\0\0\x0C\0\0\0\r\0\0\0\x0E\0\0\0\x0F\0\0\0\x13\0\0\0\x14\0\0\0\x16\0\0\0\x17\0\0\0&\0\0\0") }, unsafe { zerovec::ZeroVec::from_bytes_unchecked(b"IQ\0IR\0RU\0CM\0SD\0AF\0CN\0IR\0MN\0CN\0TR\0CN\0CC\0PK\0IN\0ME\0RO\0RU\0TR\0PK\0AF\0CN\0CN\0AU\0BN\0GB\0GF\0HK\0ID\0MO\0PA\0PF\0PH\0SR\0TH\0TW\0US\0VN\0") }, unsafe { zerovec::ZeroVec::from_bytes_unchecked(b"ArabArabCyrlArabArabArabArabArabArabArabLatnMongArabArabDevaLatnLatnLatnLatnArabArabCyrlHansHantHantHantHantHantHantHantHantHantHantHantHantHantHantHant") })
                },
                l2r: unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap::from_parts_unchecked(unsafe { zerovec::ZeroVec::from_bytes_unchecked(b"af\0am\0ar\0as\0astaz\0be\0bg\0bgcbhobn\0br\0brxbs\0ca\0cebchrcs\0cv\0cy\0da\0de\0doidsbel\0en\0es\0et\0eu\0fa\0ff\0fi\0filfo\0fr\0ga\0gd\0gl\0gu\0ha\0he\0hi\0hr\0hsbhu\0hy\0ia\0id\0ig\0is\0it\0ja\0jv\0ka\0keakgpkk\0km\0kn\0ko\0kokks\0ky\0lo\0lt\0lv\0maimi\0mk\0ml\0mn\0mnimr\0ms\0my\0ne\0nl\0nn\0no\0or\0pa\0pcmpl\0ps\0pt\0qu\0rajrm\0ro\0ru\0sa\0satsc\0sd\0si\0sk\0sl\0so\0sq\0sr\0su\0sv\0sw\0ta\0te\0tg\0th\0ti\0tk\0to\0tr\0tt\0uk\0ur\0uz\0vi\0wo\0xh\0yo\0yrlyuezh\0zu\0") }, unsafe { zerovec::ZeroVec::from_bytes_unchecked(b"ZA\0ET\0EG\0IN\0ES\0AZ\0BY\0BG\0IN\0IN\0BD\0FR\0IN\0BA\0ES\0PH\0US\0CZ\0RU\0GB\0DK\0DE\0IN\0DE\0GR\0US\0ES\0EE\0ES\0IR\0SN\0FI\0PH\0FO\0FR\0IE\0GB\0ES\0IN\0NG\0IL\0IN\0HR\0DE\0HU\0AM\x00001ID\0NG\0IS\0IT\0JP\0ID\0GE\0CV\0BR\0KZ\0KH\0IN\0KR\0IN\0IN\0KG\0LA\0LT\0LV\0IN\0NZ\0MK\0IN\0MN\0IN\0IN\0MY\0MM\0NP\0NL\0NO\0NO\0IN\0IN\0NG\0PL\0AF\0BR\0PE\0IN\0CH\0RO\0RU\0IN\0IN\0IT\0PK\0LK\0SK\0SI\0SO\0AL\0RS\0ID\0SE\0TZ\0IN\0IN\0TJ\0TH\0ET\0TM\0TO\0TR\0RU\0UA\0PK\0UZ\0VN\0SN\0ZA\0NG\0BR\0HK\0CN\0ZA\0") })
                },
                ls2r: unsafe {
                    #[allow(unused_unsafe)]
                    zerovec::ZeroMap2d::from_parts_unchecked(unsafe { zerovec::ZeroVec::from_bytes_unchecked(b"az\0en\0ff\0kk\0ky\0mn\0pa\0sd\0tg\0uz\0yuezh\0") }, unsafe { zerovec::ZeroVec::from_bytes_unchecked(b"\x01\0\0\0\x02\0\0\0\x03\0\0\0\x04\0\0\0\x06\0\0\0\x07\0\0\0\x08\0\0\0\x0B\0\0\0\x0C\0\0\0\r\0\0\0\x0E\0\0\0\x11\0\0\0") }, unsafe { zerovec::ZeroVec::from_bytes_unchecked(b"ArabShawAdlmArabArabLatnMongArabDevaKhojSindArabArabHansBopoHanbHant") }, unsafe { zerovec::ZeroVec::from_bytes_unchecked(b"IR\0GB\0GN\0CN\0CN\0TR\0CN\0PK\0IN\0IN\0IN\0PK\0AF\0CN\0TW\0TW\0TW\0") })
                },
            };
        }
        #[clippy::msrv = "1.66"]
        impl icu_provider::DataProvider<icu_locid_transform::provider::LocaleFallbackLikelySubtagsV1Marker> for $provider {
            fn load(&self, req: icu_provider::DataRequest) -> Result<icu_provider::DataResponse<icu_locid_transform::provider::LocaleFallbackLikelySubtagsV1Marker>, icu_provider::DataError> {
                if req.locale.is_empty() { Ok(icu_provider::DataResponse { payload: Some(icu_provider::DataPayload::from_static_ref(Self::SINGLETON_FALLBACK_LIKELYSUBTAGS_V1)), metadata: Default::default() }) } else { Err(icu_provider::DataErrorKind::ExtraneousLocale.with_req(<icu_locid_transform::provider::LocaleFallbackLikelySubtagsV1Marker as icu_provider::KeyedDataMarker>::KEY, req)) }
            }
        }
    };
}
