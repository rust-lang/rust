fn a() -> &[int] {
    let vec = [1, 2, 3, 4];
    let tail = match vec { //~ ERROR illegal borrow
        [_a, ..tail] => tail,
        _ => fail!(~"foo")
    };
    tail
}

fn main() {
    let tail = a();
    for tail.each |n| {
        io::println(fmt!("%d", *n));
    }
}
