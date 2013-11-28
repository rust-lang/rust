pub fn main() {
    let bar = ~3;
    let h: proc() -> int = proc() *bar;
    assert_eq!(h(), 3);
}
