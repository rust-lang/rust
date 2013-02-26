fn a() -> &[int] {
    let vec = [1, 2, 3, 4];
    let tail = match vec { //~ ERROR illegal borrow
        [_, ..tail] => tail,
        _ => fail!(~"a")
    };
    tail
}

fn b() -> &[int] {
    let vec = [1, 2, 3, 4];
    let init = match vec { //~ ERROR illegal borrow
        [..init, _] => init,
        _ => fail!(~"b")
    };
    init
}

fn c() -> &[int] {
    let vec = [1, 2, 3, 4];
    let slice = match vec { //~ ERROR illegal borrow
        [_, ..slice, _] => slice,
        _ => fail!(~"c")
    };
    slice
}

fn main() {}
