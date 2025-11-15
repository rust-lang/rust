use text_size::*;

#[test]
fn main() {
    let range = TextRange::default();
    _ = &""[range];
    _ = &String::new()[range];
}
