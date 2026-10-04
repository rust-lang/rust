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
    let x = Option(1); //~ ERROR cannot find function, tuple struct or tuple variant `Option` in this scope

    if let Option(_) = x { //~ ERROR cannot find tuple struct or tuple variant `Option` in this scope
        println!("It is OK.");
    }

    let y = Example::Ex(String::from("test"));

    if let Example(_) = y { //~ ERROR cannot find tuple struct or tuple variant `Example` in this scope
        println!("It is OK.");
    }

    let y = Void(); //~ ERROR cannot find function, tuple struct or tuple variant `Void` in this scope

    let z = ManyVariants(); //~ ERROR cannot find function, tuple struct or tuple variant `ManyVariants` in this scope
}

fn main() {}
