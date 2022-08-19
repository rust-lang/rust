#![warn(clippy::case_sensitive_file_extension_comparisons)]

use std::string::String;

struct TestStruct;

impl TestStruct {
    fn ends_with(self, arg: &str) {}
}

fn is_rust_file(filename: &str) -> bool {
    filename.ends_with(".rs")
}

fn main() {
    // std::string::String and &str should trigger the lint failure with .ext12
    let _ = String::new().ends_with(".ext12");
    let _ = "str".ends_with(".ext12");

    // The test struct should not trigger the lint failure with .ext12
    TestStruct {}.ends_with(".ext12");

    // std::string::String and &str should trigger the lint failure with .EXT12
    let _ = String::new().ends_with(".EXT12");
    let _ = "str".ends_with(".EXT12");

    // The test struct should not trigger the lint failure with .EXT12
    TestStruct {}.ends_with(".EXT12");

    // Should not trigger the lint failure with .eXT12
    let _ = String::new().ends_with(".eXT12");
    let _ = "str".ends_with(".eXT12");
    TestStruct {}.ends_with(".eXT12");

    // Should not trigger the lint failure with .EXT123 (too long)
    let _ = String::new().ends_with(".EXT123");
    let _ = "str".ends_with(".EXT123");
    TestStruct {}.ends_with(".EXT123");

    // Shouldn't fail if it doesn't start with a dot
    let _ = String::new().ends_with("a.ext");
    let _ = "str".ends_with("a.extA");
    TestStruct {}.ends_with("a.ext");
}
