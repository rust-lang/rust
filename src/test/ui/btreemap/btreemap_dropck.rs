struct PrintOnDrop<'a>(&'a str);

impl Drop for PrintOnDrop<'_> {
    fn drop(&mut self) {
        println!("printint: {}", self.0);
    }
}

use std::collections::BTreeMap;
use std::iter::FromIterator;

fn main() {
    let s = String::from("Hello World!");
    let _map = BTreeMap::from_iter([((), PrintOnDrop(&s))]);
    drop(s); //~ ERROR cannot move out of `s` because it is borrowed
}
