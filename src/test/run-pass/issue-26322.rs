macro_rules! columnline {
    () => (
        (column!(), line!())
    )
}

macro_rules! indirectcolumnline {
    () => (
        (||{ columnline!() })()
    )
}

fn main() {
    let closure = || {
        columnline!()
    };
    let iflet = if let Some(_) = Some(0) {
        columnline!()
    } else { (0, 0) };
    let cl = columnline!();
    assert_eq!(closure(), (9, 25));
    assert_eq!(iflet, (9, 28));
    assert_eq!(cl, (14, 30));
    let indirect = indirectcolumnline!();
    assert_eq!(indirect, (20, 34));
}
