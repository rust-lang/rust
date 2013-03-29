use core::io::println;

trait Trait<T> {
    fn f(&self, x: T);
}

struct Struct {
    x: int,
    y: int,
}

impl Trait<&'static str> for Struct {
    fn f(&self, x: &'static str) {
        println(~"Hi, " + x + ~"!");
    }
}

fn f(x: @Trait<&'static str>) {
    x.f("Sue");
}

fn main() {
    let a = Struct { x: 1, y: 2 };
    let b: @Trait<&'static str> = @a;
    b.f("Fred");
    let c: ~Trait<&'static str> = ~a;
    c.f("Mary");
    let d: &Trait<&'static str> = &a;
    d.f("Joe");
    f(@a);
}

