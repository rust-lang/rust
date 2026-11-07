const EXPECTED = {
    'query': 'MyForeignType::my_method',
    'others': [
        // Test case for https://github.com/rust-lang/rust/pull/96887#pullrequestreview-967154358
        // Validates that the parent path for a foreign type method is correct.
        { 'path': 'foreign_type_path::aaaaaaa::MyForeignType', 'name': 'my_method' },
    ],
};
