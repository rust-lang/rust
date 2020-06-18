use std::collections::HashMap;

fn main() {
    for _ in HashMap::new().iter().cloned() {} //~ ERROR type mismatch
    //~^ ERROR type mismatch
}
