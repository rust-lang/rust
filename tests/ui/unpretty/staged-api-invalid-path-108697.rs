// issue: rust-lang/rust#108697
// ICE: tcx.resolutions(()) is not supported for local crate -Zunpretty=mir
// on invalid module path with staged_api
//@ compile-flags: -Zunpretty=mir
//@ normalize-stderr-test: "The system cannot find the file specified." -> "No such file or directory"
#![feature(staged_api)]
#[path = "lol"]
mod foo;
//~^ ERROR couldn't read
