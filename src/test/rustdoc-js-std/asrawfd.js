// ignore-order

const QUERY = 'RawFd::as_raw_fd';

const EXPECTED = {
    'others': [
        // Reproduction test for https://github.com/rust-lang/rust/issues/78724
        // Validate that type alias methods get the correct path.
        { 'path': 'std::os::unix::io::AsRawFd', 'name': 'as_raw_fd' },
        { 'path': 'std::os::wasi::io::AsRawFd', 'name': 'as_raw_fd' },
        { 'path': 'std::os::linux::process::PidFd', 'name': 'as_raw_fd' },
        { 'path': 'std::os::unix::io::RawFd', 'name': 'as_raw_fd' },
    ],
};
