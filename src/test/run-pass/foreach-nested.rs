// -*- rust -*-

iter two() -> int {
    put 0;
    put 1;
}

fn main() {
    let vec[int] a = vec(-1, -1, -1, -1);
    let int p = 0;

    for each (int i in two()) {
        for each (int j in two()) {
            a.(p) = 10 * i + j;
            p += 1;
        }
    }

    check (a.(0) == 0);
    check (a.(1) == 1);
    check (a.(2) == 10);
    check (a.(3) == 11);
}
