trait add {
    fn plus(++x: self) -> self;
}

impl int: add {
    fn plus(++x: int) -> int { self + x }
}

fn do_add<A:add>(x: A, y: A) -> A { x.plus(y) }

fn main() {
    let x = 3 as add;
    let y = 4 as add;
    do_add(x, y); //~ ERROR a boxed trait with self types may not be passed as a bounded type
}
