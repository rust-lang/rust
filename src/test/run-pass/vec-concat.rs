


// -*- rust -*-
fn main() {
    let a: [int] = ~[1, 2, 3, 4, 5];
    let b: [int] = ~[6, 7, 8, 9, 0];
    let v: [int] = a + b;
    log v.(9);
    assert (v.(0) == 1);
    assert (v.(7) == 8);
    assert (v.(9) == 0);
}