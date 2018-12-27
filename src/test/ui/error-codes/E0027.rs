struct Dog {
    name: String,
    age: u32,
}

fn main() {
    let d = Dog { name: "Rusty".to_string(), age: 8 };

    match d {
        Dog { age: x } => {}
        //~^ ERROR pattern does not mention field `name`
    }
}
