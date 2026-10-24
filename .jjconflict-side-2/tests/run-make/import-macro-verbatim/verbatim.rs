//! Include a file by concatenating the verbatim path using `/` instead of `\`

include!(concat!(env!("VERBATIM_DIR"), "/include/include.txt"));
fn main() {
    assert_eq!(TEST, "Hello World!");

    let s = include_str!(concat!(env!("VERBATIM_DIR"), "/include/include.txt"));
    assert_eq!(s, "static TEST: &str = \"Hello World!\";\n");

    let b = include_bytes!(concat!(env!("VERBATIM_DIR"), "/include/include.txt"));
    assert_eq!(b, b"static TEST: &str = \"Hello World!\";\n");
}
