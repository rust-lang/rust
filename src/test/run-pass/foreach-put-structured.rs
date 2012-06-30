

fn pairs(it: fn((int, int))) {
    let mut i: int = 0;
    let mut j: int = 0;
    while i < 10 { it((i, j)); i += 1; j += i; }
}

fn main() {
    let mut i: int = 10;
    let mut j: int = 0;
    do pairs() |p| {
        let (_0, _1) = p;
        log(debug, _0);
        log(debug, _1);
        assert (_0 + 10 == i);
        i += 1;
        j = _1;
    };
    assert (j == 45);
}
