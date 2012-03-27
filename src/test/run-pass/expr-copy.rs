fn f(arg: {mut a: int}) {
    arg.a = 100;
}

fn main() {
    let x = {mut a: 10};
    f(x);
    assert x.a == 100;
    x.a = 20;
    f(copy x);
    assert x.a == 20;
}
