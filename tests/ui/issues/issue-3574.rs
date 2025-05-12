//@ run-pass
// rustc --test match_borrowed_str.rs.rs && ./match_borrowed_str.rs


fn compare(x: &str, y: &str) -> bool {
    match x {
        "foo" => y == "foo",
        _ => y == "bar",
    }
}

pub fn main() {
    assert!(compare("foo", "foo"));
}
