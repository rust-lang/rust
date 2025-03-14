const FILTER_CRATE = "std";
const EXPECTED = [
    {
        // Keep this test case identical to `transmute-fail`, except the
        // should-fail tag and the search query below:
        'query': 'generic:T -> generic:U',
        'others': [
            { 'path': 'std::intrinsics::simd', 'name': 'simd_as' },
            { 'path': 'std::intrinsics::simd', 'name': 'simd_cast' },
            { 'path': 'std::mem', 'name': 'transmute' },
        ],
    },
];
