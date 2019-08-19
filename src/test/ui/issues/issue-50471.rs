// build-pass (FIXME(62277): could be check-pass?)

fn main() {
    assert!({false});

    assert!(r"\u{41}" == "A");

    assert!(r"\u{".is_empty());
}
