//@ run-rustfix
use std::collections::HashMap;

fn main() {
    let vs = vec![0, 0, 1, 1, 3, 4, 5, 6, 3, 3, 3];

    let mut counts = HashMap::new();
    for num in vs {
        let count = counts.entry(num).or_insert(0);
        *count += 1;
    }

    let _ = counts.iter().max_by_key(|(_, v)| v);
    //~^ ERROR lifetime may not live long enough
    //~| HELP dereference the return value
}
