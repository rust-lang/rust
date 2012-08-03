trait add {
    fn plus(x: self) -> self;
}

fn do_add(x: add, y: add) -> add {
    x.plus(y) //~ ERROR can not call a method that contains a self type through a boxed trait
}

fn main() {}
