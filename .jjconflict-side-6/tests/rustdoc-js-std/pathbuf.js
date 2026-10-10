// The PathBuf type is defined in std,
// but used in proc_macro. This means both crates'
// search indexes contain TypeData for it,
// but only std defines EntryData.
// This test case ensures we can merge them.
//
// https://github.com/rust-lang/rust/issues/162334


const EXPECTED = [
    {
        query: 'PathBuf',
        others: [
            { 'path': 'std::path', 'name': 'PathBuf' },
        ],
    },
];
