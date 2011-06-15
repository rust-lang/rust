


// -*- rust -*-
iter two() -> int { put 0; put 1; }

iter range(int start, int stop) -> int {
    let int i = start;
    while (i < stop) { put i; i += 1; }
}

fn main() {
    let vec[mutable int] a = [mutable -1, -1, -1, -1, -1, -1, -1, -1];
    let int p = 0;
    for each (int i in two()) {
        for each (int j in range(0, 2)) {
            let int tmp = 10 * i + j;
            for each (int k in range(0, 2)) { a.(p) = 10 * tmp + k; p += 1; }
        }
    }
    assert (a.(0) == 0);
    assert (a.(1) == 1);
    assert (a.(2) == 10);
    assert (a.(3) == 11);
    assert (a.(4) == 100);
    assert (a.(5) == 101);
    assert (a.(6) == 110);
    assert (a.(7) == 111);
}