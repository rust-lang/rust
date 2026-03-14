// When mutably indexing a type that implements `Index` but not `IndexMut`, a
// special 'help' message is added to the output.
//
// The suggestions have different requirements on the index type.
// map[&idx] = val;
// If idx is later used, it results in a borrow  issue if both:
// - the the suggestion that was chosen requires the idx, not a ref to it.
// - idx's  type is not Copy.
//
// For now, we suggest all options, regardless of the Copy-ness of the idx.
use std::collections::HashMap;

/// With a copy type, subsequent reuse of idx is not an issue.
fn copy_type() {
    // ===&str===, a copy type.

    let mut map = HashMap::<&str, u32>::new();
    // earlier, peter is initialised with 22.
    // map["peter"] = 22; //ERROR
    map.insert("peter", 22);

    // at some point, if we get a &str variable peter again
    let peter = "peter";
    // and we want to use it to update the map but still want to use it later?
    // map[&peter] = 23; // ERROR
    // we could insert again, and because &T are copy, we can use peter even if we use peter later.
    map.insert(peter, 23); // WORKS
    println!("my name is {peter}"); // WORKS because &str is Copy
    // and we could use a &&str too in this case, because &str:Borrow<&str> (because T:Borrow<T>)
    if let Some(val) = map.get_mut(&peter) {
        *val = 23;
    }; // WORKS
    println!("my name is {peter}"); // WORKS because &str is Copy
    // even a &str directly, (because rust auto borrows peter -> &peter ?)
    if let Some(val) = map.get_mut(peter) {
        *val = 24;
    }; // WORKS
}

/// With a non-copy type, subsequent reuse of idx is an issue for `insert` and `entry`.
fn non_copy_type_insert() {
    // ===STRING===, a non-copy type

    let mut map = HashMap::<String, u32>::new();
    // earlier, peter is initialised with 22.
    // map[&"peter".to_string()] = 22; // ERROR cannot assign
    map.insert("peter".to_string(), 22);

    // at some point, if we get a String variable peter again
    let peter = "peter".to_string();
    // and we want to use it to update the map but still want to use it later?
    // map[&peter] = 23; // ERROR cannot assign
    // we could insert again, but we cannot use peter after.
    map.insert(peter, 23); // WORKS
    println!("my name is {peter}"); //~ ERROR: borrow of moved value: `peter` [E0382]
}

/// With a non-copy type, subsequent reuse of idx is not an issue for `get_mut`.
fn non_copy_type_get_mut() {
    // ===STRING===, a non-copy type

    let mut map = HashMap::<String, u32>::new();
    // earlier, peter is initialised with 22.
    // map["peter".to_string()] = 22; // ERROR cannot assign
    map.insert("peter".to_string(), 22);

    // at some point, if we get a String variable peter again
    let peter = "peter".to_string();
    // and we want to use it to update the map but still want to use it later?
    // map[&peter] = 23; // ERROR cannot assign
    // we can use a &String in this case, so get_mut is always fine.
    if let Some(val) = map.get_mut(&peter) {
        *val = 23;
    }; // WORKS
    println!("my name is {peter}"); // WORKS
    // or a &str because String:Borrow<str>) and "peter" is &str.

    if let Some(val) = map.get_mut("peter") {
        *val = 24;
    }; // WORKS
    println!("my name is {peter}"); // WORKS
}

fn main() {}
