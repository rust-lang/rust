#![no_main]
#[macro_use] extern crate libfuzzer_sys;
extern crate ra_syntax;

fuzz_target!(|data: &[u8]| {
    if let Ok(text) = std::str::from_utf8(data) {
        ra_syntax::utils::check_fuzz_invariants(text)
    }
});
