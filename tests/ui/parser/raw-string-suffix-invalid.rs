// Test for issue #144161: Wrong error for string literal suffix when stuck together

fn main() {
    let s = r#" \\ "#r"\\ ";
    //~^ ERROR suffixes on string literals are invalid
}
