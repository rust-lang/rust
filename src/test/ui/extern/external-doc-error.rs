// normalize-stderr-test: "The system cannot find the file specified\." -> "No such file or directory"
// ignore-tidy-linelength

#![feature(external_doc)]

#[doc(include = "not-a-file.md")] //~ ERROR: couldn't read
pub struct SomeStruct;

fn main() {}
