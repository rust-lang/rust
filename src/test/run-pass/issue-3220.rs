struct thing { x: int, }

impl thing : Drop {
    fn finalize() {}
}

fn thing() -> thing {
    thing {
        x: 0
    }
}
impl thing { fn f(self) {} }

fn main() {
    let z = thing();
    (move z).f();
}
