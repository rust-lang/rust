#![crate_type = "rlib"]
//@ edition: 2024
//@ compile-flags: -Zunpretty=thir-flat
//@ check-pass

// Snapshot test capturing the THIR pattern structure produced by
// string-literal and string-constant patterns.

pub fn hello_world(x: &str) {
    match x {
        "hello" => {}
        CONSTANT => {}
        _ => {}
    }
}

const CONSTANT: &str = "constant";
