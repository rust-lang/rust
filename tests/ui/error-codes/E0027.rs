struct Dog {
    name: String,
    age: u32,
}


fn main() {
    let d = Dog { name: "Rusty".to_string(), age: 8 };

    match d {
        Dog { age: x } => {} //~ ERROR pattern does not mention field `name`
    }
    match d {
        // trailing comma
        Dog { name: x, } => {} //~ ERROR pattern does not mention field `age`
    }
    match d {
        // trailing comma with weird whitespace
        Dog { name: x  , } => {} //~ ERROR pattern does not mention field `age`
    }
    match d {
        Dog {} => {} //~ ERROR pattern does not mention fields `name`, `age`
    }
}
