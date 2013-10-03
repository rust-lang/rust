#[allow(dead_assignment)];
#[allow(unused_variable)];
#[feature(struct_variant)];

enum Animal {
    Dog (~str, f64),
    Cat { name: ~str, weight: f64 }
}

pub fn main() {
    let mut a: Animal = Dog(~"Cocoa", 37.2);
    a = Cat{ name: ~"Spotty", weight: 2.7 };
    // permuting the fields should work too
    let _c = Cat { weight: 3.1, name: ~"Spreckles" };
}
