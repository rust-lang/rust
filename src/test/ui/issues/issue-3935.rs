// run-pass

#[derive(PartialEq)]
struct Bike {
    name: String,
}

pub fn main() {
    let town_bike = Bike { name: "schwinn".to_string() };
    let my_bike = Bike { name: "surly".to_string() };

    assert!(town_bike != my_bike);
}
