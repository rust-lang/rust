// exact-check
// ignore-order
// ignore-tidy-linelength

// This test case verifies that the href points at the correct impl

const FILTER_CRATE = "std";

const EXPECTED = [
    {
        'query': 'simd<i16>, simd<i16> -> simd<i16>',
        'others': [
            {
                'path': 'std::simd::Simd',
                'name': 'simd_max',
                'href': '../std/simd/struct.Simd.html#impl-SimdOrd-for-Simd%3Ci16,+N%3E/method.simd_max'
            },
            {
                'path': 'std::simd::Simd',
                'name': 'simd_min',
                'href': '../std/simd/struct.Simd.html#impl-SimdOrd-for-Simd%3Ci16,+N%3E/method.simd_min'
            },
            {
                'path': 'std::simd::Simd',
                'name': 'saturating_add',
                'href': '../std/simd/struct.Simd.html#impl-SimdInt-for-Simd%3Ci16,+N%3E/method.saturating_add'
            },
            {
                'path': 'std::simd::Simd',
                'name': 'saturating_sub',
                'href': '../std/simd/struct.Simd.html#impl-SimdInt-for-Simd%3Ci16,+N%3E/method.saturating_sub'
            },
            {
                'path': 'std::simd::Simd',
                'name': 'simd_clamp',
                'href': '../std/simd/struct.Simd.html#impl-SimdOrd-for-Simd%3Ci16,+N%3E/method.simd_clamp'
            },
        ],
    },
    {
        'query': 'simd<i8>, simd<i8> -> simd<i8>',
        'others': [
            {
                'path': 'std::simd::Simd',
                'name': 'simd_max',
                'href': '../std/simd/struct.Simd.html#impl-SimdOrd-for-Simd%3Ci8,+N%3E/method.simd_max'
            },
            {
                'path': 'std::simd::Simd',
                'name': 'simd_min',
                'href': '../std/simd/struct.Simd.html#impl-SimdOrd-for-Simd%3Ci8,+N%3E/method.simd_min'
            },
            {
                'path': 'std::simd::Simd',
                'name': 'saturating_add',
                'href': '../std/simd/struct.Simd.html#impl-SimdInt-for-Simd%3Ci8,+N%3E/method.saturating_add'
            },
            {
                'path': 'std::simd::Simd',
                'name': 'saturating_sub',
                'href': '../std/simd/struct.Simd.html#impl-SimdInt-for-Simd%3Ci8,+N%3E/method.saturating_sub'
            },
            {
                'path': 'std::simd::Simd',
                'name': 'simd_clamp',
                'href': '../std/simd/struct.Simd.html#impl-SimdOrd-for-Simd%3Ci8,+N%3E/method.simd_clamp'
            },
        ],
    },
];
