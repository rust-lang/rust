iface foo {
    fn bar(x: uint) -> self;
}
impl of foo for int {
    fn bar() -> int {
        self
    }
}

fn main() {
}
