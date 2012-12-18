fn a() -> &int {
    let vec = [1, 2, 3, 4];
    let tail = match vec {
        [_a, ..tail] => &tail[0], //~ ERROR illegal borrow
        _ => fail ~"foo"
    };
    move tail
}

fn main() {
    let fifth = a();
    io::println(fmt!("%d", *fifth));
}
