const EXPECTED = [
    {
        query: 'PathBuf',
        others: [
            // ensure hashset::insert comes first
            { 'path': 'std::path', 'name': 'PathBuf' },
        ],
    },
];
