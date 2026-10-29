struct Person {
    first_name: String,
    age: u32,
}

fn first_woman(man: &Person) -> Person {
    Person {
        first_name: "Eve".to_string(),
        ..man.clone() //~ ERROR: mismatched types
    }
}

fn main() {
    let adam = Person {
        first_name: "Adam".to_string(),
        age: 0,
    };

    let eve = first_woman(&adam);
}
