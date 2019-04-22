use std::collections::HashMap;
fn main() {
    let things: HashMap<String, Vec<String>> = HashMap::new();
    for src in things.keys() {
        things[src.as_str()].sort(); //~ ERROR cannot borrow data in a `&` reference as mutable
    }
}
