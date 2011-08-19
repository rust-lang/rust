


// -*- rust -*-
iter two() -> int { put 0; put 1; }

iter range(start: int, stop: int) -> int {
    let i: int = start;
    while i < stop { put i; i += 1; }
}

fn main() {
    let a: [mutable int] = [mutable -1, -1, -1, -1, -1, -1, -1, -1];
    let p: int = 0;
    for each i: int in two() {
        for each j: int in range(0, 2) {
            let tmp: int = 10 * i + j;
            for each k: int in range(0, 2) { a[p] = 10 * tmp + k; p += 1; }
        }
    }
    assert (a[0] == 0);
    assert (a[1] == 1);
    assert (a[2] == 10);
    assert (a[3] == 11);
    assert (a[4] == 100);
    assert (a[5] == 101);
    assert (a[6] == 110);
    assert (a[7] == 111);
}
