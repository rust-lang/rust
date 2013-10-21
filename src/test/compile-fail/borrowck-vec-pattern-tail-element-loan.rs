fn a() -> &int {
    let vec = ~[1, 2, 3, 4];
    let tail = match vec {
        [_a, ..tail] => &tail[0], //~ ERROR borrowed value does not live long enough
        _ => fail!("foo")
    };
    tail
}

fn main() {
    let fifth = a();
    println!("{}", *fifth);
}
