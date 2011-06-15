


// -*- rust -*-
fn main() {
    let vec[int] v = [10, 20];
    assert (v.(0) == 10);
    assert (v.(1) == 20);
    let int x = 0;
    assert (v.(x) == 10);
    assert (v.(x + 1) == 20);
    x = x + 1;
    assert (v.(x) == 20);
    assert (v.(x - 1) == 10);
}