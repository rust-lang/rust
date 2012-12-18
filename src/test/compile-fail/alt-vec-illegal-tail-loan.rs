// xfail-test

fn a() -> &[int] {
    let vec = [1, 2, 3, 4];
    let tail = match vec {
        [_a, ..tail] => tail, //~ ERROR illegal borrow
        _ => fail ~"foo"
    };
    move tail
}

fn main() {
    let tail = a();
    for tail.each |n| {
        io::println(fmt!("%d", *n));
    }
}
