// run-pass

use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn main() {
    let m: HashMap<PathBuf, ()> = HashMap::new();
    let k = Path::new("foo");
    println!("{:?}", m.get(k));
}
