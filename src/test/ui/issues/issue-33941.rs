// compile-flags: -Zdeduplicate-diagnostics=yes

use std::collections::HashMap;

fn main() {
    for _ in HashMap::new().iter().cloned() {} //~ ERROR expected `std::collections::hash_map::Iter<'_, _, _>` to be an iterator that yields `&_`, but it yields `(&_, &_)`
    //~^ ERROR expected `std::collections::hash_map::Iter<'_, _, _>` to be an iterator that yields `&_`, but it yields `(&_, &_)`
    //~| ERROR expected `std::collections::hash_map::Iter<'_, _, _>` to be an iterator that yields `&_`, but it yields `(&_, &_)`
}
