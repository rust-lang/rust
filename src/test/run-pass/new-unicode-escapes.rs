pub fn main() {
    let s = "\u{2603}";
    assert_eq!(s, "☃");

    let s = "\u{2a10}\u{2A01}\u{2Aa0}";
    assert_eq!(s, "⨐⨁⪠");

    let s = "\\{20}";
    let mut correct_s = String::from("\\");
    correct_s.push_str("{20}");
    assert_eq!(s, correct_s);
}
