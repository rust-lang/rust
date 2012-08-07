trait add {
    fn plus(x: self) -> self;
}

fn do_add(x: add, y: add) -> add {
    x.plus(y) //~ ERROR cannot call a method whose type contains a self-type through a boxed trait
}

fn main() {}
