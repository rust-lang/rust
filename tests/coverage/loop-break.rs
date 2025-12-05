//@ edition: 2021

fn main() {
    loop {
        if core::hint::black_box(true) {
            break;
        }
    }
}

// This test is a lightly-modified version of `tests/mir-opt/coverage/instrument_coverage.rs`.
// If this test needs to be blessed, then the mir-opt version probably needs to
// be blessed too!
