

iter pairs() -> {_0: int, _1: int} {
    let i: int = 0;
    let j: int = 0;
    while i < 10 { put {_0: i, _1: j}; i += 1; j += i; }
}

fn main() {
    let i: int = 10;
    let j: int = 0;
    for each p: {_0: int, _1: int}  in pairs() {
        log p._0;
        log p._1;
        assert (p._0 + 10 == i);
        i += 1;
        j = p._1;
    }
    assert (j == 45);
}