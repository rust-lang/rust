
const tau: float = 2.0*3.14159265358979323;

type point = {x: float, y: float};
type size = {w: float, h: float};
enum shape {
    circle(point, float),
    rectangle(point, size)
}


fn compute_area(shape: &shape) -> float {
    match *shape {
        circle(_, radius) => 0.5 * tau * radius * radius,
        rectangle(_, ref size) => size.w * size.h
    }
}

impl shape {
    // self is in the implicit self region
    fn select<T>(&self, threshold: float,
                 a: &r/T, b: &r/T) -> &r/T {
        if compute_area(self) > threshold {a} else {b}
    }
}

fn select_based_on_unit_circle<T>(
    threshold: float, a: &r/T, b: &r/T) -> &r/T {

    let shape = &circle({x: 0.0, y: 0.0}, 1.0);
    shape.select(threshold, a, b)
}


struct thing {
    x: {mut a: @int};
}

fn thing(x: {mut a: @int}) -> thing {
    thing {
        x: copy x
    }
}

impl thing {
    fn foo(@self) -> int { *self.x.a }
    fn bar(~self) -> int { *self.x.a }
    fn quux(&self) -> int { *self.x.a }
    fn baz(&self) -> &self/{mut a: @int} { &self.x }
    fn spam(self) -> int { *self.x.a }
}

trait Nus { fn f(&self); }
impl thing: Nus { fn f(&self) {} }

fn main() {

    let x = @thing({mut a: @10});
    assert x.foo() == 10;
    assert x.quux() == 10;

    let y = ~thing({mut a: @10});
    assert y.bar() == 10;
    assert y.quux() == 10;

    let z = thing({mut a: @11});
    assert z.spam() == 11;
}
