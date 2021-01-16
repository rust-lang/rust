// run-pass

pub fn main() {
    let x = 1;
    let y = 2;

    assert_eq!(3, match (x, y) {
        (1, 1) => 1,
        (2, 2) => 2,
        (1..=2, 2) => 3,
        _ => 4,
    });

    // nested tuple
    assert_eq!(3, match ((x, y),) {
        ((1, 1),) => 1,
        ((2, 2),) => 2,
        ((1..=2, 2),) => 3,
        _ => 4,
    });
}
