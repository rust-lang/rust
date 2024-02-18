#![warn(clippy::incompatible_msrv)]
#![feature(custom_inner_attributes)]
#![clippy::msrv = "1.3.0"]

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::thread::sleep;
use std::time::Duration;

fn foo() {
    let mut map: HashMap<&str, u32> = HashMap::new();
    assert_eq!(map.entry("poneyland").key(), &"poneyland");
    //~^ ERROR: is `1.3.0` but this item is stable since `1.10.0`
    if let Entry::Vacant(v) = map.entry("poneyland") {
        v.into_key();
        //~^ ERROR: is `1.3.0` but this item is stable since `1.12.0`
    }
    // Should warn for `sleep` but not for `Duration` (which was added in `1.3.0`).
    sleep(Duration::new(1, 0));
    //~^ ERROR: is `1.3.0` but this item is stable since `1.4.0`
}

fn main() {}
