// run-pass
#![allow(dead_code)]
#![allow(unused_assignments)]
// pretty-expanded FIXME #23616

#![allow(unused_variables)]

enum Animal {
    Dog (String, f64),
    Cat { name: String, weight: f64 }
}

pub fn main() {
    let mut a: Animal = Animal::Dog("Cocoa".to_string(), 37.2);
    a = Animal::Cat{ name: "Spotty".to_string(), weight: 2.7 };
    // permuting the fields should work too
    let _c = Animal::Cat { weight: 3.1, name: "Spreckles".to_string() };
}
