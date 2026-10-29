// Regression test for https://github.com/rust-lang/rust/issues/115466

// The "identifier" in format strings is parsed as an IDENTIFIER_OR_KEYWORD, not an IDENTIFIER.
// Test that there is an actionable diagnostic if a RAW_IDENTIFIER is used instead.

fn main() {
    let r#type = "foobar";
    println!("It is {r#type}"); //~ ERROR: invalid format string: raw identifiers are not supported
    println!(r##"It still is {r#type}"##); //~ ERROR: invalid format string: raw identifiers are not supported
    println!(concat!("{r#", "type}")); //~ ERROR: invalid format string: raw identifiers are not supported
    println!("{\x72\x23type:?}"); //~ ERROR: invalid format string: raw identifiers are not supported

    // OK
    println!("{type}");
    println!("{let}", let = r#type);
    println!("{let}", r#let = r#type);
}
