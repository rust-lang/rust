

iter pairs() -> rec(int _0, int _1) {
    let int i = 0;
    let int j = 0;
    while (i < 10) { put rec(_0=i, _1=j); i += 1; j += i; }
}

fn main() {
    let int i = 10;
    let int j = 0;
    for each (rec(int _0, int _1) p in pairs()) {
        log p._0;
        log p._1;
        assert (p._0 + 10 == i);
        i += 1;
        j = p._1;
    }
    assert (j == 45);
}