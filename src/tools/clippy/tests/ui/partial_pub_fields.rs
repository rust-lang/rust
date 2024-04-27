#![allow(unused)]
#![warn(clippy::partial_pub_fields)]

fn main() {
    use std::collections::HashMap;

    #[derive(Default)]
    pub struct FileSet {
        files: HashMap<String, u32>,
        pub paths: HashMap<u32, String>,
        //~^ ERROR: mixed usage of pub and non-pub fields
    }

    pub struct Color {
        pub r: u8,
        pub g: u8,
        b: u8,
        //~^ ERROR: mixed usage of pub and non-pub fields
    }

    pub struct Point(i32, pub i32);
    //~^ ERROR: mixed usage of pub and non-pub fields

    pub struct Visibility {
        r#pub: bool,
        pub pos: u32,
        //~^ ERROR: mixed usage of pub and non-pub fields
    }

    // Don't lint on empty structs;
    pub struct Empty1;
    pub struct Empty2();
    pub struct Empty3 {};

    // Don't lint on structs with one field.
    pub struct Single1(i32);
    pub struct Single2(pub i32);
    pub struct Single3 {
        v1: i32,
    }
    pub struct Single4 {
        pub v1: i32,
    }
}
