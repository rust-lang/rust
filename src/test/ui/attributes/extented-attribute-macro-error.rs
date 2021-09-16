#![feature(extended_key_value_attributes)]
#![doc = include_str!("../not_existing_file.md")]
struct Documented {}
//~^^ ERROR No such file or directory

fn main() {}
