enum Fruit {
    Apple(String, String),
    Pear(u32),
    Orange((String, String)),
    Banana(()),
}

fn main() {
    let x = Fruit::Apple(String::new(), String::new());
    match x {
        Fruit::Apple(a) => {}, //~ ERROR E0023
        Fruit::Apple(a, b, c) => {}, //~ ERROR E0023
        Fruit::Pear(1, 2) => {}, //~ ERROR E0023
        Fruit::Orange(a, b) => {}, //~ ERROR E0023
        Fruit::Banana() => {}, //~ ERROR E0023
    }
}
