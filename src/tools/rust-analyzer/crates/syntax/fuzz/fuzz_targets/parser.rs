//! Fuzzing for from-scratch parsing.

#![no_main]
use libfuzzer_sys::fuzz_target;
use syntax::fuzz::check_parser;

fuzz_target!(|data: &[u8]| {
    if let Ok(text) = std::str::from_utf8(data) {
        check_parser(text)
    }
});
