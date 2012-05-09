iface add {
    fn +(x: self) -> self;
}

impl of add for int {
    fn +(x: int) -> int { self + x }
}

impl of add for @const int {
    fn +(x: @const int) -> @const int { @(*self + *x) }
}

fn do_add<A:add>(+x: A, +y: A) -> A { x + y }

fn main() {
    assert do_add(3, 4) == 7;
    assert do_add(@3, @4) == @7;
    assert do_add(@mut 3, @mut 4) == @mut 7;
}
