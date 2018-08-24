#![feature(unicode_version)]

/// Tests access to the internal Unicode Version type and value.
pub fn main() {
    check(std::char::UNICODE_VERSION);
}

pub fn check(unicode_version: std::char::UnicodeVersion) {
    assert!(unicode_version.major >= 10);
}
