struct thing { x: int; drop { } }
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
