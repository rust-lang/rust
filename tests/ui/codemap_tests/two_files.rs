include!("two_files_data.rs");

struct Baz { }

impl Bar for Baz { } //~ ERROR expected trait, found type alias

fn main() { }
