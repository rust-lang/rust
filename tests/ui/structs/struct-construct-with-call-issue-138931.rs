struct PersonOnlyName {
    name: String
}

struct PersonWithAge {
    name: String,
    age: u8,
    height: u8,
}



fn main() {
    let wilfred = PersonOnlyName("Name1".to_owned());
    //~^ ERROR expected function, tuple struct or tuple variant, found struct `PersonOnlyName` [E0423]

    let bill = PersonWithAge( //~ ERROR expected function, tuple struct or tuple variant, found struct `PersonWithAge` [E0423]
        "Name2".to_owned(),
        20,
        180,
    );

    let person = PersonWithAge("Name3".to_owned());
    //~^ ERROR expected function, tuple struct or tuple variant, found struct `PersonWithAge` [E0423]
}
