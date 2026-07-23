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
    //~^ ERROR cannot find function, tuple struct or tuple variant `PersonOnlyName` in this scope [E0423]

    let bill = PersonWithAge( //~ ERROR cannot find function, tuple struct or tuple variant `PersonWithAge` in this scope [E0423]
        "Name2".to_owned(),
        20,
        180,
    );

    let person = PersonWithAge("Name3".to_owned());
    //~^ ERROR cannot find function, tuple struct or tuple variant `PersonWithAge` in this scope [E0423]
}
