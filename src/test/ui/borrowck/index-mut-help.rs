// When mutably indexing a type that implements `Index` but not `IndexMut`, a
// special 'help' message is added to the output.


fn main() {
    use std::collections::HashMap;

    let mut map = HashMap::new();
    map.insert("peter", "23".to_string());

    map["peter"].clear();           //~ ERROR
    map["peter"] = "0".to_string(); //~ ERROR
    let _ = &mut map["peter"];      //~ ERROR
}
