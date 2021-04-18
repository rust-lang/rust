// run-pass
// Regression test for issue #48508:
//
// Confusion between global and local file offsets caused incorrect handling of multibyte character
// spans when compiling multiple files. One visible effect was an ICE generating debug information
// when a multibyte character is at the end of a scope. The problematic code is actually in
// issue-48508-aux.rs

// compile-flags:-g
// ignore-pretty issue #37195
// ignore-asmjs wasm2js does not support source maps yet

#![allow(uncommon_codepoints)]

#[path = "issue-48508-aux.rs"]
mod other_file;

fn main() {
    other_file::other();
}
