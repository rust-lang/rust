
const x: &int = &10;

const y: &{a: int, b: &int} = &{a: 15, b: x};

fn main() {
    io::println(fmt!("x = %?", *x));
    io::println(fmt!("y = {a: %?, b: %?}", y.a, *(y.b)));
    assert *x == 10;
    assert *(y.b) == 10;
}
