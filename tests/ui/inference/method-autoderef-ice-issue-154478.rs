//@ compile-flags: -Znext-solver=globally

#![feature(arbitrary_self_types_pointers)]

fn main() {
    let mut map = std::collections::HashMap::new();
    map.insert(0, "string".to_owned());

    let string = &map[&0];
    string.push_str(13);
    //~^ ERROR mismatched types
}
