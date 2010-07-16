// -*- rust -*-

iter two() -> int {
    put 0;
    put 1;
}

iter range(int start, int stop) -> int {
    let int i = start;
    while (i < stop) {
        put i;
        i += 1;
    }
}

fn main() {
    let vec[int] a = vec(-1, -1, -1, -1, -1, -1, -1, -1);
    let int p = 0;

    for each (int i in two()) {
        for each (int j in range(0, 2)) {
            let int tmp = 10 * i + j;
            for each (int k in range(0, 2)) {
                a.(p) = 10 * tmp + k;
                p += 1;
            }
        }
    }

    check (a.(0) == 0);
    check (a.(1) == 1);
    check (a.(2) == 10);
    check (a.(3) == 11);
    check (a.(4) == 100);
    check (a.(5) == 101);
    check (a.(6) == 110);
    check (a.(7) == 111);
}
