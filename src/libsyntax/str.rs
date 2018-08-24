#[inline]
pub fn char_at(s: &str, byte: usize) -> char {
    s[byte..].chars().next().unwrap()
}
