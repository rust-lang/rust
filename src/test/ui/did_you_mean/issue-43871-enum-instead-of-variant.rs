enum Example { Ex(String), NotEx }

enum Void {}

enum ManyVariants {
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
    Ten,
}

fn result_test() {
    let x = Option(1); //~ ERROR expected function, found enum

    if let Option(_) = x { //~ ERROR expected tuple struct/variant, found enum
        println!("It is OK.");
    }

    let y = Example::Ex(String::from("test"));

    if let Example(_) = y { //~ ERROR expected tuple struct/variant, found enum
        println!("It is OK.");
    }

    let y = Void(); //~ ERROR expected function, found enum

    let z = ManyVariants(); //~ ERROR expected function, found enum
}

fn main() {}
