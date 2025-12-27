// Test for raw string literals with invalid suffixes

fn main() {
    // Raw string literal with hash delimiters and suffix
    let s2 = r##"test"##suffix;
    //~^ ERROR suffixes on string literals are invalid
}
