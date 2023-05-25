// exact-match

// https://github.com/rust-lang/rust/issues/60485#issuecomment-663900624
const QUERY = 'OsString -> String';

const EXPECTED = {
    'others': [
        { 'path': 'std::ffi::OsString', 'name': 'into_string' },
    ]
};
