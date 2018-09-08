#![no_main]
#[macro_use] extern crate libfuzzer_sys;
extern crate libsyntax2;

fuzz_target!(|data: &[u8]| {
    if let Ok(text) = std::str::from_utf8(data) {
        libsyntax2::utils::check_fuzz_invariants(text)
    }
});
