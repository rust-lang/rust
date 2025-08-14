const FILTER_CRATE = "core";
const EXPECTED = [
    {
        'query': 'generic:T -> generic:U',
        'others': [
            { 'path': 'core::intrinsics::simd', 'name': 'simd_as' },
            { 'path': 'core::intrinsics::simd', 'name': 'simd_cast' },
            { 'path': 'core::mem', 'name': 'transmute' },
        ],
    },
];
