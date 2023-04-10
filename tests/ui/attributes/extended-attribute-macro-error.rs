// normalize-stderr-test: "couldn't read.*" -> "couldn't read the file"

#![doc = include_str!("../nonexistent_file.md")]
struct Documented {}
//~^^ ERROR couldn't read

fn main() {}
