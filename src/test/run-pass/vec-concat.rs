


// -*- rust -*-
fn main() {
    let vec[int] a = [1, 2, 3, 4, 5];
    let vec[int] b = [6, 7, 8, 9, 0];
    let vec[int] v = a + b;
    log v.(9);
    assert (v.(0) == 1);
    assert (v.(7) == 8);
    assert (v.(9) == 0);
}