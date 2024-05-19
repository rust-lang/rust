//@ run-pass

fn main() {
    let mut escaped = String::from("");
    for c in '\u{10401}'.escape_unicode() {
        escaped.push(c);
    }
    assert_eq!("\\u{10401}", escaped);
}
