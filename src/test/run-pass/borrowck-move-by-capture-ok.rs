pub fn main() {
    let bar = ~3;
    let h: @fn() -> int = || *bar;
    assert_eq!(h(), 3);
}
