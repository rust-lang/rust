// Test for raw string literal with word suffix

fn main() {
    // Raw string literal with word suffix
    let s3 = r#"test"#invalid;
    //~^ ERROR suffixes on string literals are invalid
}
