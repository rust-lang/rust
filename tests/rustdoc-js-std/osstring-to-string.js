// exact-match

// https://github.com/rust-lang/rust/issues/60485#issuecomment-663900624
const EXPECTED = {
    'query': 'OsString -> String',
    'others': [
        { 'path': 'std::ffi::OsString', 'name': 'into_string' },
    ]
};
