enum Example { Ex(String), NotEx }

fn result_test() {
    let x = Option(1); //~ ERROR expected function, found enum

    if let Option(_) = x { //~ ERROR expected tuple struct/variant, found enum
        println!("It is OK.");
    }

    let y = Example::Ex(String::from("test"));

    if let Example(_) = y { //~ ERROR expected tuple struct/variant, found enum
        println!("It is OK.");
    }
}

fn main() {}
