#![warn(clippy::verbose_file_reads)]
use std::env::temp_dir;
use std::fs::File;
use std::io::Read;

struct Struct;
// To make sure we only warn on File::{read_to_end, read_to_string} calls
impl Struct {
    pub fn read_to_end(&self) {}

    pub fn read_to_string(&self) {}
}

fn main() -> std::io::Result<()> {
    let path = "foo.txt";
    // Lint shouldn't catch this
    let s = Struct;
    s.read_to_end();
    s.read_to_string();
    // Should catch this
    let mut f = File::open(path)?;
    let mut buffer = Vec::new();
    f.read_to_end(&mut buffer)?;
    // ...and this
    let mut string_buffer = String::new();
    f.read_to_string(&mut string_buffer)?;
    Ok(())
}
