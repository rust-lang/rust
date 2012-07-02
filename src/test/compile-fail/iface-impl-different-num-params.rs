iface foo {
    fn bar(x: uint) -> self;
}
impl of foo for int {
    fn bar() -> int {
        //~^ ERROR method `bar` has 0 parameters but the iface has 1
        self
    }
}

fn main() {
}
