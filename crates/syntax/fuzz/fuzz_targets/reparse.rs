//! FIXME: write short doc here

#![no_main]
use libfuzzer_sys::fuzz_target;
use syntax::fuzz::CheckReparse;

fuzz_target!(|data: &[u8]| {
    if let Some(check) = CheckReparse::from_data(data) {
        check.run();
    }
});
