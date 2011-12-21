fn to_lambda(f: lambda(uint) -> uint) -> lambda(uint) -> uint {
    ret f;
}

fn main() {
    let x: lambda(uint) -> uint = to_lambda({ |x| x * 2u });
    let y = to_lambda(x);

    let x_r = x(22u);
    let y_r = y(x_r);

    assert x_r == 44u;
    assert y_r == 88u;
}
