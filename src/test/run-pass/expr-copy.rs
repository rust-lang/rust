fn f(arg: {mutable a: int}) {
    arg.a = 100;
}

fn main() {
    let x = {mutable a: 10};
    f(x);
    assert x.a == 100;
    x.a = 20;
    f(copy x);
    assert x.a == 20;
}
