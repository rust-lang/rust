// run-pass

#![feature(unicode_version)]

/// Tests access to the internal Unicode Version type and value.
pub fn main() {
    check(std::char::UNICODE_VERSION);
}

pub fn check(unicode_version: (u8, u8, u8)) {
    assert!(unicode_version.0 >= 10);
}
