// run-pass
// ignore-wasm32-bare FIXME(#93923) llvm miscompilation

use std::collections::HashMap;
use std::path::Path;

fn main() {
    let mut map = HashMap::new();
    map.insert(Path::new("a"), 0);
    map.get(Path::new("a"));
}
