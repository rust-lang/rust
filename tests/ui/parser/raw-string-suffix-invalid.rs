// Test for issue #144161: Wrong error for string literal suffix when stuck together
// This tests the case where a raw string has an invalid suffix immediately followed by another string

fn main() {
    let s = r#" \\ "#r"\\ ";
    //~^ ERROR suffixes on string literals are invalid
    //~| ERROR expected one of `.`, `;`, `?`, `else`, or an operator
}
