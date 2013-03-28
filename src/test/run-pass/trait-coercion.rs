use core::io::println;

trait Trait {
    fn f(&self);
}

struct Struct {
    x: int,
    y: int,
}

impl Trait for Struct {
    fn f(&self) {
        println("Hi!");
    }
}

fn f(x: @Trait) {
    x.f();
}

fn main() {
    let a = Struct { x: 1, y: 2 };
    let b: @Trait = @a;
    b.f();
    let c: ~Trait = ~a;
    c.f();
    let d: &Trait = &a;
    d.f();
    f(@a);
}

