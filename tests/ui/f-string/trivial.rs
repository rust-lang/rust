// run-pass

pub fn main() {
    assert_eq!(f"", "");
    assert_eq!(f"foo", "foo");
    assert_eq!(f"a{{b}}c", "a{b}c");
    assert_eq!(f"a\{b\}c", "a{b}c");
}
