// normalize-stderr-test: "not-a-file.md:.*\(" -> "not-a-file.md: $$FILE_NOT_FOUND_MSG ("

#![feature(external_doc)]

#[doc(include = "not-a-file.md")] //~ ERROR: couldn't read
pub struct SomeStruct;

fn main() {}
