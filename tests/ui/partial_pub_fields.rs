#![allow(unused)]
#![warn(clippy::partial_pub_fields)]

fn main() {
    // test code goes here

    use std::collections::HashMap;

    #[derive(Default)]
    pub struct FileSet {
        files: HashMap<String, u32>,
        pub paths: HashMap<u32, String>,
    }

    pub struct Color {
        pub r: u8,
        pub g: u8,
        b: u8,
    }
}
