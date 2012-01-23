fn test(f: fn(uint) -> uint) -> uint {
    ret f(22u);
}

fn main() {
    let y = test(fn~(x: uint) -> uint { ret 4u * x; });
    assert y == 88u;
}