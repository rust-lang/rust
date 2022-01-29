// run-pass
use std::collections::HashMap;

pub fn main() {
    let mut m = HashMap::new();
    m.insert(b"foo".to_vec(), b"bar".to_vec());
    println!("{:?}", m);
}
