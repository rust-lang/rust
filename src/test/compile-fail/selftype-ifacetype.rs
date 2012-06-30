iface add {
    fn +(x: self) -> self;
}

fn do_add(x: add, y: add) -> add {
    x + y //~ ERROR can not call a method that contains a self type through a boxed iface
}

fn main() {}
