//@ run-pass
#![allow(unused_variables)]

enum Homura {
    Madoka {
        name: String,
        age: u32,
    },
}

fn main() {
    let homura = Homura::Madoka {
        name: "Akemi".to_string(),
        age: 14,
    };

    match homura {
        Homura::Madoka {
            name,
            age,
        } => (),
    };
}
