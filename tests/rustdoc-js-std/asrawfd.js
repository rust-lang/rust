// ignore-order

const EXPECTED = {
    'query': 'method:RawFd::as_raw_fd',
    'others': [
        // Reproduction test for https://github.com/rust-lang/rust/issues/78724
        // Validate that type alias methods get the correct path.
        { 'path': 'std::os::fd::RawFd', 'name': 'as_raw_fd' },
    ],
};
