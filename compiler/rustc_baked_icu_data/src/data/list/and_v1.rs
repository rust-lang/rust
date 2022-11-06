// @generated
type DataStruct = <::icu_list::provider::AndListV1Marker as ::icu_provider::DataMarker>::Yokeable;
pub static DATA: litemap::LiteMap<&str, &DataStruct, &[(&str, &DataStruct)]> =
    litemap::LiteMap::from_sorted_store_unchecked(&[
        ("en", EN),
        ("es", ES),
        ("fr", FR),
        ("it", IT),
        ("ja", JA),
        ("pt", PT),
        ("ru", RU),
        ("tr", TR),
        ("und", UND),
        ("zh", ZH_ZH_HANS),
        ("zh-Hans", ZH_ZH_HANS),
        ("zh-Hant", ZH_HANT),
    ]);
static EN: &DataStruct = &::icu_list::provider::ListFormatterPatternsV1([
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", and ", 6u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" and ", 5u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", & ", 4u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" & ", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
]);
static ES: &DataStruct = &::icu_list::provider::ListFormatterPatternsV1([
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" y ", 3u8)
        },
        special_case: Some(::icu_list::provider::SpecialCasePattern {
            condition: unsafe {
                ::icu_list::provider::StringMatcher::from_dfa_bytes_unchecked(&[
                    114u8, 117u8, 115u8, 116u8, 45u8, 114u8, 101u8, 103u8, 101u8, 120u8, 45u8,
                    97u8, 117u8, 116u8, 111u8, 109u8, 97u8, 116u8, 97u8, 45u8, 100u8, 102u8, 97u8,
                    45u8, 115u8, 112u8, 97u8, 114u8, 115u8, 101u8, 0u8, 0u8, 255u8, 254u8, 0u8,
                    0u8, 2u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 14u8, 0u8, 0u8, 0u8, 1u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 1u8, 2u8, 2u8, 2u8, 3u8, 4u8, 4u8, 5u8, 6u8, 7u8, 7u8, 7u8, 7u8,
                    7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8,
                    7u8, 7u8, 7u8, 8u8, 9u8, 9u8, 9u8, 10u8, 11u8, 11u8, 12u8, 13u8, 14u8, 14u8,
                    14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8,
                    14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8,
                    15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 16u8, 16u8, 16u8,
                    16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8,
                    17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8,
                    17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8,
                    17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 18u8, 18u8, 19u8, 19u8, 19u8, 19u8, 19u8,
                    19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8,
                    19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 20u8,
                    21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 22u8,
                    23u8, 23u8, 24u8, 25u8, 25u8, 25u8, 26u8, 27u8, 27u8, 27u8, 27u8, 27u8, 27u8,
                    27u8, 27u8, 27u8, 27u8, 27u8, 40u8, 1u8, 0u8, 0u8, 1u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 1u8, 128u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 5u8,
                    0u8, 5u8, 5u8, 6u8, 6u8, 12u8, 12u8, 13u8, 13u8, 0u8, 0u8, 83u8, 0u8, 0u8, 0u8,
                    68u8, 0u8, 0u8, 0u8, 83u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 2u8, 0u8, 0u8, 27u8, 0u8, 0u8, 18u8, 0u8, 0u8, 0u8, 18u8, 0u8, 0u8,
                    0u8, 0u8, 3u8, 0u8, 6u8, 6u8, 13u8, 13u8, 0u8, 0u8, 104u8, 0u8, 0u8, 0u8,
                    104u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 14u8, 0u8, 0u8, 0u8, 2u8, 2u8,
                    4u8, 7u8, 9u8, 9u8, 11u8, 14u8, 19u8, 19u8, 20u8, 20u8, 21u8, 21u8, 22u8, 22u8,
                    23u8, 23u8, 24u8, 24u8, 25u8, 25u8, 26u8, 26u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8,
                    68u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8,
                    0u8, 191u8, 0u8, 0u8, 0u8, 206u8, 0u8, 0u8, 0u8, 221u8, 0u8, 0u8, 0u8, 236u8,
                    0u8, 0u8, 0u8, 221u8, 0u8, 0u8, 0u8, 251u8, 0u8, 0u8, 0u8, 10u8, 1u8, 0u8, 0u8,
                    25u8, 1u8, 0u8, 0u8, 18u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 17u8, 0u8, 0u8,
                    68u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 17u8, 17u8, 0u8, 0u8,
                    191u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 17u8, 0u8, 0u8,
                    191u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 16u8, 0u8, 0u8,
                    191u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 16u8, 17u8, 0u8, 0u8,
                    221u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 17u8, 0u8, 0u8,
                    221u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 15u8, 0u8, 0u8,
                    221u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 4u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 35u8,
                    0u8, 0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 9u8, 0u8, 0u8, 0u8, 18u8, 0u8, 0u8, 0u8,
                    18u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 35u8, 0u8, 0u8,
                    0u8, 35u8, 0u8, 0u8, 0u8,
                ])
            },
            pattern: unsafe {
                ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" e ", 3u8)
            },
        }),
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" y ", 3u8)
        },
        special_case: Some(::icu_list::provider::SpecialCasePattern {
            condition: unsafe {
                ::icu_list::provider::StringMatcher::from_dfa_bytes_unchecked(&[
                    114u8, 117u8, 115u8, 116u8, 45u8, 114u8, 101u8, 103u8, 101u8, 120u8, 45u8,
                    97u8, 117u8, 116u8, 111u8, 109u8, 97u8, 116u8, 97u8, 45u8, 100u8, 102u8, 97u8,
                    45u8, 115u8, 112u8, 97u8, 114u8, 115u8, 101u8, 0u8, 0u8, 255u8, 254u8, 0u8,
                    0u8, 2u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 14u8, 0u8, 0u8, 0u8, 1u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 1u8, 2u8, 2u8, 2u8, 3u8, 4u8, 4u8, 5u8, 6u8, 7u8, 7u8, 7u8, 7u8,
                    7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8,
                    7u8, 7u8, 7u8, 8u8, 9u8, 9u8, 9u8, 10u8, 11u8, 11u8, 12u8, 13u8, 14u8, 14u8,
                    14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8,
                    14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8,
                    15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 16u8, 16u8, 16u8,
                    16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8,
                    17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8,
                    17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8,
                    17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 18u8, 18u8, 19u8, 19u8, 19u8, 19u8, 19u8,
                    19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8,
                    19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 20u8,
                    21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 22u8,
                    23u8, 23u8, 24u8, 25u8, 25u8, 25u8, 26u8, 27u8, 27u8, 27u8, 27u8, 27u8, 27u8,
                    27u8, 27u8, 27u8, 27u8, 27u8, 40u8, 1u8, 0u8, 0u8, 1u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 1u8, 128u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 5u8,
                    0u8, 5u8, 5u8, 6u8, 6u8, 12u8, 12u8, 13u8, 13u8, 0u8, 0u8, 83u8, 0u8, 0u8, 0u8,
                    68u8, 0u8, 0u8, 0u8, 83u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 2u8, 0u8, 0u8, 27u8, 0u8, 0u8, 18u8, 0u8, 0u8, 0u8, 18u8, 0u8, 0u8,
                    0u8, 0u8, 3u8, 0u8, 6u8, 6u8, 13u8, 13u8, 0u8, 0u8, 104u8, 0u8, 0u8, 0u8,
                    104u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 14u8, 0u8, 0u8, 0u8, 2u8, 2u8,
                    4u8, 7u8, 9u8, 9u8, 11u8, 14u8, 19u8, 19u8, 20u8, 20u8, 21u8, 21u8, 22u8, 22u8,
                    23u8, 23u8, 24u8, 24u8, 25u8, 25u8, 26u8, 26u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8,
                    68u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8,
                    0u8, 191u8, 0u8, 0u8, 0u8, 206u8, 0u8, 0u8, 0u8, 221u8, 0u8, 0u8, 0u8, 236u8,
                    0u8, 0u8, 0u8, 221u8, 0u8, 0u8, 0u8, 251u8, 0u8, 0u8, 0u8, 10u8, 1u8, 0u8, 0u8,
                    25u8, 1u8, 0u8, 0u8, 18u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 17u8, 0u8, 0u8,
                    68u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 17u8, 17u8, 0u8, 0u8,
                    191u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 17u8, 0u8, 0u8,
                    191u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 16u8, 0u8, 0u8,
                    191u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 16u8, 17u8, 0u8, 0u8,
                    221u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 17u8, 0u8, 0u8,
                    221u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 15u8, 0u8, 0u8,
                    221u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 4u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 35u8,
                    0u8, 0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 9u8, 0u8, 0u8, 0u8, 18u8, 0u8, 0u8, 0u8,
                    18u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 35u8, 0u8, 0u8,
                    0u8, 35u8, 0u8, 0u8, 0u8,
                ])
            },
            pattern: unsafe {
                ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" e ", 3u8)
            },
        }),
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" y ", 3u8)
        },
        special_case: Some(::icu_list::provider::SpecialCasePattern {
            condition: unsafe {
                ::icu_list::provider::StringMatcher::from_dfa_bytes_unchecked(&[
                    114u8, 117u8, 115u8, 116u8, 45u8, 114u8, 101u8, 103u8, 101u8, 120u8, 45u8,
                    97u8, 117u8, 116u8, 111u8, 109u8, 97u8, 116u8, 97u8, 45u8, 100u8, 102u8, 97u8,
                    45u8, 115u8, 112u8, 97u8, 114u8, 115u8, 101u8, 0u8, 0u8, 255u8, 254u8, 0u8,
                    0u8, 2u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 14u8, 0u8, 0u8, 0u8, 1u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 1u8, 2u8, 2u8, 2u8, 3u8, 4u8, 4u8, 5u8, 6u8, 7u8, 7u8, 7u8, 7u8,
                    7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8,
                    7u8, 7u8, 7u8, 8u8, 9u8, 9u8, 9u8, 10u8, 11u8, 11u8, 12u8, 13u8, 14u8, 14u8,
                    14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8,
                    14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8,
                    15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 16u8, 16u8, 16u8,
                    16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8,
                    17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8,
                    17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8,
                    17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 18u8, 18u8, 19u8, 19u8, 19u8, 19u8, 19u8,
                    19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8,
                    19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 20u8,
                    21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 22u8,
                    23u8, 23u8, 24u8, 25u8, 25u8, 25u8, 26u8, 27u8, 27u8, 27u8, 27u8, 27u8, 27u8,
                    27u8, 27u8, 27u8, 27u8, 27u8, 40u8, 1u8, 0u8, 0u8, 1u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 1u8, 128u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 5u8,
                    0u8, 5u8, 5u8, 6u8, 6u8, 12u8, 12u8, 13u8, 13u8, 0u8, 0u8, 83u8, 0u8, 0u8, 0u8,
                    68u8, 0u8, 0u8, 0u8, 83u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 2u8, 0u8, 0u8, 27u8, 0u8, 0u8, 18u8, 0u8, 0u8, 0u8, 18u8, 0u8, 0u8,
                    0u8, 0u8, 3u8, 0u8, 6u8, 6u8, 13u8, 13u8, 0u8, 0u8, 104u8, 0u8, 0u8, 0u8,
                    104u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 14u8, 0u8, 0u8, 0u8, 2u8, 2u8,
                    4u8, 7u8, 9u8, 9u8, 11u8, 14u8, 19u8, 19u8, 20u8, 20u8, 21u8, 21u8, 22u8, 22u8,
                    23u8, 23u8, 24u8, 24u8, 25u8, 25u8, 26u8, 26u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8,
                    68u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8,
                    0u8, 191u8, 0u8, 0u8, 0u8, 206u8, 0u8, 0u8, 0u8, 221u8, 0u8, 0u8, 0u8, 236u8,
                    0u8, 0u8, 0u8, 221u8, 0u8, 0u8, 0u8, 251u8, 0u8, 0u8, 0u8, 10u8, 1u8, 0u8, 0u8,
                    25u8, 1u8, 0u8, 0u8, 18u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 17u8, 0u8, 0u8,
                    68u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 17u8, 17u8, 0u8, 0u8,
                    191u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 17u8, 0u8, 0u8,
                    191u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 16u8, 0u8, 0u8,
                    191u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 16u8, 17u8, 0u8, 0u8,
                    221u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 17u8, 0u8, 0u8,
                    221u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 15u8, 0u8, 0u8,
                    221u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 4u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 35u8,
                    0u8, 0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 9u8, 0u8, 0u8, 0u8, 18u8, 0u8, 0u8, 0u8,
                    18u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 35u8, 0u8, 0u8,
                    0u8, 35u8, 0u8, 0u8, 0u8,
                ])
            },
            pattern: unsafe {
                ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" e ", 3u8)
            },
        }),
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" y ", 3u8)
        },
        special_case: Some(::icu_list::provider::SpecialCasePattern {
            condition: unsafe {
                ::icu_list::provider::StringMatcher::from_dfa_bytes_unchecked(&[
                    114u8, 117u8, 115u8, 116u8, 45u8, 114u8, 101u8, 103u8, 101u8, 120u8, 45u8,
                    97u8, 117u8, 116u8, 111u8, 109u8, 97u8, 116u8, 97u8, 45u8, 100u8, 102u8, 97u8,
                    45u8, 115u8, 112u8, 97u8, 114u8, 115u8, 101u8, 0u8, 0u8, 255u8, 254u8, 0u8,
                    0u8, 2u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 14u8, 0u8, 0u8, 0u8, 1u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 1u8, 2u8, 2u8, 2u8, 3u8, 4u8, 4u8, 5u8, 6u8, 7u8, 7u8, 7u8, 7u8,
                    7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8,
                    7u8, 7u8, 7u8, 8u8, 9u8, 9u8, 9u8, 10u8, 11u8, 11u8, 12u8, 13u8, 14u8, 14u8,
                    14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8,
                    14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8,
                    15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 16u8, 16u8, 16u8,
                    16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8,
                    17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8,
                    17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8,
                    17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 18u8, 18u8, 19u8, 19u8, 19u8, 19u8, 19u8,
                    19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8,
                    19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 20u8,
                    21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 22u8,
                    23u8, 23u8, 24u8, 25u8, 25u8, 25u8, 26u8, 27u8, 27u8, 27u8, 27u8, 27u8, 27u8,
                    27u8, 27u8, 27u8, 27u8, 27u8, 40u8, 1u8, 0u8, 0u8, 1u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 1u8, 128u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 5u8,
                    0u8, 5u8, 5u8, 6u8, 6u8, 12u8, 12u8, 13u8, 13u8, 0u8, 0u8, 83u8, 0u8, 0u8, 0u8,
                    68u8, 0u8, 0u8, 0u8, 83u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 2u8, 0u8, 0u8, 27u8, 0u8, 0u8, 18u8, 0u8, 0u8, 0u8, 18u8, 0u8, 0u8,
                    0u8, 0u8, 3u8, 0u8, 6u8, 6u8, 13u8, 13u8, 0u8, 0u8, 104u8, 0u8, 0u8, 0u8,
                    104u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 14u8, 0u8, 0u8, 0u8, 2u8, 2u8,
                    4u8, 7u8, 9u8, 9u8, 11u8, 14u8, 19u8, 19u8, 20u8, 20u8, 21u8, 21u8, 22u8, 22u8,
                    23u8, 23u8, 24u8, 24u8, 25u8, 25u8, 26u8, 26u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8,
                    68u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8,
                    0u8, 191u8, 0u8, 0u8, 0u8, 206u8, 0u8, 0u8, 0u8, 221u8, 0u8, 0u8, 0u8, 236u8,
                    0u8, 0u8, 0u8, 221u8, 0u8, 0u8, 0u8, 251u8, 0u8, 0u8, 0u8, 10u8, 1u8, 0u8, 0u8,
                    25u8, 1u8, 0u8, 0u8, 18u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 17u8, 0u8, 0u8,
                    68u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 17u8, 17u8, 0u8, 0u8,
                    191u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 17u8, 0u8, 0u8,
                    191u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 16u8, 0u8, 0u8,
                    191u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 16u8, 17u8, 0u8, 0u8,
                    221u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 17u8, 0u8, 0u8,
                    221u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 15u8, 0u8, 0u8,
                    221u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 4u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 35u8,
                    0u8, 0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 9u8, 0u8, 0u8, 0u8, 18u8, 0u8, 0u8, 0u8,
                    18u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 35u8, 0u8, 0u8,
                    0u8, 35u8, 0u8, 0u8, 0u8,
                ])
            },
            pattern: unsafe {
                ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" e ", 3u8)
            },
        }),
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" y ", 3u8)
        },
        special_case: Some(::icu_list::provider::SpecialCasePattern {
            condition: unsafe {
                ::icu_list::provider::StringMatcher::from_dfa_bytes_unchecked(&[
                    114u8, 117u8, 115u8, 116u8, 45u8, 114u8, 101u8, 103u8, 101u8, 120u8, 45u8,
                    97u8, 117u8, 116u8, 111u8, 109u8, 97u8, 116u8, 97u8, 45u8, 100u8, 102u8, 97u8,
                    45u8, 115u8, 112u8, 97u8, 114u8, 115u8, 101u8, 0u8, 0u8, 255u8, 254u8, 0u8,
                    0u8, 2u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 14u8, 0u8, 0u8, 0u8, 1u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 1u8, 2u8, 2u8, 2u8, 3u8, 4u8, 4u8, 5u8, 6u8, 7u8, 7u8, 7u8, 7u8,
                    7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8,
                    7u8, 7u8, 7u8, 8u8, 9u8, 9u8, 9u8, 10u8, 11u8, 11u8, 12u8, 13u8, 14u8, 14u8,
                    14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8,
                    14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8,
                    15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 16u8, 16u8, 16u8,
                    16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8,
                    17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8,
                    17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8,
                    17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 18u8, 18u8, 19u8, 19u8, 19u8, 19u8, 19u8,
                    19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8,
                    19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 20u8,
                    21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 22u8,
                    23u8, 23u8, 24u8, 25u8, 25u8, 25u8, 26u8, 27u8, 27u8, 27u8, 27u8, 27u8, 27u8,
                    27u8, 27u8, 27u8, 27u8, 27u8, 40u8, 1u8, 0u8, 0u8, 1u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 1u8, 128u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 5u8,
                    0u8, 5u8, 5u8, 6u8, 6u8, 12u8, 12u8, 13u8, 13u8, 0u8, 0u8, 83u8, 0u8, 0u8, 0u8,
                    68u8, 0u8, 0u8, 0u8, 83u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 2u8, 0u8, 0u8, 27u8, 0u8, 0u8, 18u8, 0u8, 0u8, 0u8, 18u8, 0u8, 0u8,
                    0u8, 0u8, 3u8, 0u8, 6u8, 6u8, 13u8, 13u8, 0u8, 0u8, 104u8, 0u8, 0u8, 0u8,
                    104u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 14u8, 0u8, 0u8, 0u8, 2u8, 2u8,
                    4u8, 7u8, 9u8, 9u8, 11u8, 14u8, 19u8, 19u8, 20u8, 20u8, 21u8, 21u8, 22u8, 22u8,
                    23u8, 23u8, 24u8, 24u8, 25u8, 25u8, 26u8, 26u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8,
                    68u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8,
                    0u8, 191u8, 0u8, 0u8, 0u8, 206u8, 0u8, 0u8, 0u8, 221u8, 0u8, 0u8, 0u8, 236u8,
                    0u8, 0u8, 0u8, 221u8, 0u8, 0u8, 0u8, 251u8, 0u8, 0u8, 0u8, 10u8, 1u8, 0u8, 0u8,
                    25u8, 1u8, 0u8, 0u8, 18u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 17u8, 0u8, 0u8,
                    68u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 17u8, 17u8, 0u8, 0u8,
                    191u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 17u8, 0u8, 0u8,
                    191u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 16u8, 0u8, 0u8,
                    191u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 16u8, 17u8, 0u8, 0u8,
                    221u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 17u8, 0u8, 0u8,
                    221u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 15u8, 0u8, 0u8,
                    221u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 4u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 35u8,
                    0u8, 0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 9u8, 0u8, 0u8, 0u8, 18u8, 0u8, 0u8, 0u8,
                    18u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 35u8, 0u8, 0u8,
                    0u8, 35u8, 0u8, 0u8, 0u8,
                ])
            },
            pattern: unsafe {
                ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" e ", 3u8)
            },
        }),
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" y ", 3u8)
        },
        special_case: Some(::icu_list::provider::SpecialCasePattern {
            condition: unsafe {
                ::icu_list::provider::StringMatcher::from_dfa_bytes_unchecked(&[
                    114u8, 117u8, 115u8, 116u8, 45u8, 114u8, 101u8, 103u8, 101u8, 120u8, 45u8,
                    97u8, 117u8, 116u8, 111u8, 109u8, 97u8, 116u8, 97u8, 45u8, 100u8, 102u8, 97u8,
                    45u8, 115u8, 112u8, 97u8, 114u8, 115u8, 101u8, 0u8, 0u8, 255u8, 254u8, 0u8,
                    0u8, 2u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 14u8, 0u8, 0u8, 0u8, 1u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 1u8, 2u8, 2u8, 2u8, 3u8, 4u8, 4u8, 5u8, 6u8, 7u8, 7u8, 7u8, 7u8,
                    7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8, 7u8,
                    7u8, 7u8, 7u8, 8u8, 9u8, 9u8, 9u8, 10u8, 11u8, 11u8, 12u8, 13u8, 14u8, 14u8,
                    14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8,
                    14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 14u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8,
                    15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 15u8, 16u8, 16u8, 16u8,
                    16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8, 16u8,
                    17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8,
                    17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 17u8,
                    17u8, 17u8, 17u8, 17u8, 17u8, 17u8, 18u8, 18u8, 19u8, 19u8, 19u8, 19u8, 19u8,
                    19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8,
                    19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 19u8, 20u8,
                    21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 21u8, 22u8,
                    23u8, 23u8, 24u8, 25u8, 25u8, 25u8, 26u8, 27u8, 27u8, 27u8, 27u8, 27u8, 27u8,
                    27u8, 27u8, 27u8, 27u8, 27u8, 40u8, 1u8, 0u8, 0u8, 1u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 0u8, 0u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 1u8, 128u8,
                    0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 1u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 5u8,
                    0u8, 5u8, 5u8, 6u8, 6u8, 12u8, 12u8, 13u8, 13u8, 0u8, 0u8, 83u8, 0u8, 0u8, 0u8,
                    68u8, 0u8, 0u8, 0u8, 83u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 2u8, 0u8, 0u8, 27u8, 0u8, 0u8, 18u8, 0u8, 0u8, 0u8, 18u8, 0u8, 0u8,
                    0u8, 0u8, 3u8, 0u8, 6u8, 6u8, 13u8, 13u8, 0u8, 0u8, 104u8, 0u8, 0u8, 0u8,
                    104u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 14u8, 0u8, 0u8, 0u8, 2u8, 2u8,
                    4u8, 7u8, 9u8, 9u8, 11u8, 14u8, 19u8, 19u8, 20u8, 20u8, 21u8, 21u8, 22u8, 22u8,
                    23u8, 23u8, 24u8, 24u8, 25u8, 25u8, 26u8, 26u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8,
                    68u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8, 0u8, 68u8, 0u8, 0u8,
                    0u8, 191u8, 0u8, 0u8, 0u8, 206u8, 0u8, 0u8, 0u8, 221u8, 0u8, 0u8, 0u8, 236u8,
                    0u8, 0u8, 0u8, 221u8, 0u8, 0u8, 0u8, 251u8, 0u8, 0u8, 0u8, 10u8, 1u8, 0u8, 0u8,
                    25u8, 1u8, 0u8, 0u8, 18u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 17u8, 0u8, 0u8,
                    68u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 17u8, 17u8, 0u8, 0u8,
                    191u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 17u8, 0u8, 0u8,
                    191u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 16u8, 0u8, 0u8,
                    191u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 16u8, 17u8, 0u8, 0u8,
                    221u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 17u8, 0u8, 0u8,
                    221u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 2u8, 0u8, 15u8, 15u8, 0u8, 0u8,
                    221u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 4u8, 0u8, 0u8, 0u8, 0u8, 0u8,
                    0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 35u8,
                    0u8, 0u8, 0u8, 35u8, 0u8, 0u8, 0u8, 9u8, 0u8, 0u8, 0u8, 18u8, 0u8, 0u8, 0u8,
                    18u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 0u8, 35u8, 0u8, 0u8,
                    0u8, 35u8, 0u8, 0u8, 0u8,
                ])
            },
            pattern: unsafe {
                ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" e ", 3u8)
            },
        }),
    },
]);
static FR: &DataStruct = &::icu_list::provider::ListFormatterPatternsV1([
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" et ", 4u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" et ", 4u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" et ", 4u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" et ", 4u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
]);
static IT: &DataStruct = &::icu_list::provider::ListFormatterPatternsV1([
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" e ", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" e ", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" e ", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" e ", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" e ", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" e ", 3u8)
        },
        special_case: None,
    },
]);
static JA: &DataStruct = &::icu_list::provider::ListFormatterPatternsV1([
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
]);
static PT: &DataStruct = &::icu_list::provider::ListFormatterPatternsV1([
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" e ", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" e ", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" e ", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" e ", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
]);
static RU: &DataStruct = &::icu_list::provider::ListFormatterPatternsV1([
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" и ", 4u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" и ", 4u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" и ", 4u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" и ", 4u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
]);
static TR: &DataStruct = &::icu_list::provider::ListFormatterPatternsV1([
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" ve ", 4u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" ve ", 4u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" ve ", 4u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(" ve ", 4u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
]);
static UND: &DataStruct = &::icu_list::provider::ListFormatterPatternsV1([
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked(", ", 2u8)
        },
        special_case: None,
    },
]);
static ZH_HANT: &DataStruct = &::icu_list::provider::ListFormatterPatternsV1([
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("和", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("和", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("和", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("和", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("和", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("和", 3u8)
        },
        special_case: None,
    },
]);
static ZH_ZH_HANS: &DataStruct = &::icu_list::provider::ListFormatterPatternsV1([
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("和", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("和", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("和", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("和", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
    ::icu_list::provider::ConditionalListJoinerPattern {
        default: unsafe {
            ::icu_list::provider::ListJoinerPattern::from_parts_unchecked("、", 3u8)
        },
        special_case: None,
    },
]);
