// Checks that `.` is considered the same as `::`.

const EXPECTED = [
    {
        'query': 'Vec.new',
        'others': [
            { 'path': 'std::vec::Vec', 'name': 'new' },
            { 'path': 'std::vec::Vec', 'name': 'new_in' },
        ],
    },
    {
        'query': 'ffi.string.new',
        'others': [
            { 'path': 'std::ffi::CString', 'name': 'new' },
        ],
    },
];
