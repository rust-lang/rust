#[allow(dead_assignment)];
#[allow(unused_variable)];

enum Animal {
    Dog (~str, float),
    Cat { name: ~str, weight: float }
}

pub fn main() {
    let mut a: Animal = Dog(~"Cocoa", 37.2);
    a = Cat{ name: ~"Spotty", weight: 2.7 };
    // permuting the fields should work too
    let _c = Cat { weight: 3.1, name: ~"Spreckles" };
}
