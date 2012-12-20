enum Animal {
    Dog (~str, float),
    Cat { name: ~str, weight: float }
}

fn main() {
    let mut a: Animal = Dog(~"Cocoa", 37.2);
    a = Cat{ name: ~"Spotty", weight: 2.7 };
    // permuting the fields should work too
    let c = Cat { weight: 3.1, name: ~"Spreckles" };
}
