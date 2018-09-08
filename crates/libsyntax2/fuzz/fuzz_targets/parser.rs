#![no_main]
#[macro_use] extern crate libfuzzer_sys;
extern crate libsyntax2;

fuzz_target!(|data: &[u8]| {
    if let Ok(text) = std::str::from_utf8(data) {
        let x = libsyntax2::File::parse(text);
        let _ = x.ast();
        let _ = x.syntax();
        let _ = x.errors();
    }
});
