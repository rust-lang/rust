//@ run-pass

use std::collections::HashMap;

fn process_or_insert_default(map: &mut HashMap<usize, String>, key: usize) {
    match map.get_mut(&key) {
        Some(value) => {
            process(value);
        }
        None => {
            map.insert(key, "".to_string());
        }
    }
}

fn process(x: &str) {
    assert_eq!(x, "Hello, world");
}

fn main() {
    let map = &mut HashMap::new();
    map.insert(22, format!("Hello, world"));
    map.insert(44, format!("Goodbye, world"));
    process_or_insert_default(map, 22);
    process_or_insert_default(map, 66);
    assert_eq!(map[&66], "");
}
