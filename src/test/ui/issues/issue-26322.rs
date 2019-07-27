// run-pass
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

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
    assert_eq!(closure(), (9, 19));
    assert_eq!(iflet, (9, 22));
    assert_eq!(cl, (14, 24));
    let indirect = indirectcolumnline!();
    assert_eq!(indirect, (20, 28));
}
