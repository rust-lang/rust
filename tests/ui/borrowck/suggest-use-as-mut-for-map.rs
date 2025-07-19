// We don't suggest change `&` to `&mut`
// instead we suggest using .get_mut() instead of &mut, see issue #143732
fn main() {
    let mut map = std::collections::BTreeMap::new();
    map.insert(0, "string".to_owned());

    let string = &map[&0];
    string.push_str("test"); //~ ERROR cannot borrow `*string` as mutable, as it is behind a `&` reference [E0596]

    let mut map = std::collections::HashMap::new();
    map.insert(0, "string".to_owned());

    let string = &map[&0];
    string.push_str("test"); //~ ERROR cannot borrow `*string` as mutable, as it is behind a `&` reference [E0596]
}
