// Regression Test for ICE: https://github.com/rust-lang/rust/issues/150263
//
// This one's about the MIR layout checker being too strict. When you project
// a field from a multi-variant enum (like accessing `some_result.0`), the
// compiler checks if the resulting type can fit in registers or needs to be
// stored in memory.
//
// Problem was, the old code assumed field projections would ALWAYS result in
// types small enough for registers. But with enums containing big stuff like
// `Box<dyn Read>`, that's not true - they still need memory storage. The
// compiler would hit a debug assertion and crash instead of just using memory.
//
//@ build-pass
//@ compile-flags: -C opt-level=3

#![allow(unused)]

use std::io::Read;

// Enums with chunky payloads that need memory storage
enum GetResultPayload {
    File(Box<dyn Read>),
    Stream(Vec<u8>),
}

// Another complex enum to trigger the layout issue
enum ComplexResult<T> {
    Ok(T),
    Err(Box<dyn std::error::Error>),
}

// Function that projects fields from multi-variant enums
// This triggers MIR analysis that checks layout properties
fn process_result(result: GetResultPayload) {
    match result {
        GetResultPayload::File(f) => {
            // Field projection on multi-variant enum
            let _reader = f;
        }
        GetResultPayload::Stream(s) => {
            let _data = s;
        }
    }
}

fn process_complex<T>(result: ComplexResult<T>) {
    match result {
        ComplexResult::Ok(val) => {
            // Projection that might result in backend-ref layout
            let _value = val;
        }
        ComplexResult::Err(e) => {
            let _error = e;
        }
    }
}

fn main() {
    let payload = GetResultPayload::Stream(vec![1, 2, 3]);
    process_result(payload);

    let complex: ComplexResult<String> = ComplexResult::Ok("test".to_string());
    process_complex(complex);
}
