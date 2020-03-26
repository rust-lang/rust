use text_size::*;

#[test]
fn main() {
    let range = TextRange::default();
    &""[range];
    &String::new()[range];
}
