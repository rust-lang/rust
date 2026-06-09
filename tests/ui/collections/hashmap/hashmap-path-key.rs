//! regression test for <https://github.com/rust-lang/rust/issues/23036>
//@ run-pass

use std::collections::HashMap;
use std::path::Path;

fn main() {
    let mut map = HashMap::new();
    map.insert(Path::new("a"), 0);
    map.get(Path::new("a"));
}
