struct thing { x: int; new () { self.x = 0; } drop { } }
impl thing { fn f(self) {} }

fn main() {
    let z = thing();
    (move z).f();
}
