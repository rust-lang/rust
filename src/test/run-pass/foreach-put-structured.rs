

iter pairs() -> tup(int, int) {
    let int i = 0;
    let int j = 0;
    while (i < 10) { put tup(i, j); i += 1; j += i; }
}

fn main() {
    let int i = 10;
    let int j = 0;
    for each (tup(int, int) p in pairs()) {
        log p._0;
        log p._1;
        assert (p._0 + 10 == i);
        i += 1;
        j = p._1;
    }
    assert (j == 45);
}