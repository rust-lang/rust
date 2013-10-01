fn a() -> &[int] {
    let vec = ~[1, 2, 3, 4];
    let tail = match vec {
        [_, ..tail] => tail, //~ ERROR does not live long enough
        _ => fail2!("a")
    };
    tail
}

fn b() -> &[int] {
    let vec = ~[1, 2, 3, 4];
    let init = match vec {
        [..init, _] => init, //~ ERROR does not live long enough
        _ => fail2!("b")
    };
    init
}

fn c() -> &[int] {
    let vec = ~[1, 2, 3, 4];
    let slice = match vec {
        [_, ..slice, _] => slice, //~ ERROR does not live long enough
        _ => fail2!("c")
    };
    slice
}

fn main() {}
