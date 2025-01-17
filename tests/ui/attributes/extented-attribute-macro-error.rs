//@ normalize-stderr: "couldn't read.*" -> "couldn't read the file"

#![doc = include_str!("../not_existing_file.md")]
struct Documented {}
//~^^ ERROR couldn't read

fn main() {}
