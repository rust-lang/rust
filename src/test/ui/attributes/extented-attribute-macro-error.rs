// normalize-stderr-test: "couldn't read.*" -> "couldn't read the file"

#![feature(extended_key_value_attributes)]
#![doc = include_str!("../not_existing_file.md")]
struct Documented {}
//~^^ ERROR couldn't read

fn main() {}
