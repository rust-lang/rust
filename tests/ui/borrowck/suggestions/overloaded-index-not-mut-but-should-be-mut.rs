//@ run-rustfix
fn main() {
    let mut map = std::collections::BTreeMap::new();
    map.insert(0, "string".to_owned());

    let string = &map[&0];
    string.push_str("test");
    //~^ ERROR cannot borrow `*string` as mutable, as it is behind a `&` reference

    let mut map = std::collections::HashMap::new();
    map.insert(0, "string".to_owned());

    let string = &map[&0];
    string.push_str("test");
    //~^ ERROR cannot borrow `*string` as mutable, as it is behind a `&` reference

    let mut vec = vec![String::new(), String::new()];
    let string = &vec[0];
    string.push_str("test");
    //~^ ERROR cannot borrow `*string` as mutable, as it is behind a `&` reference
}
