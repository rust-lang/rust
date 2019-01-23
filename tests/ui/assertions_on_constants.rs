fn main() {
    assert!(true);
    assert!(false);
    assert!(true, "true message");
    assert!(false, "false message");

    const B: bool = true;
    assert!(B);

    const C: bool = false;
    assert!(C);
}
