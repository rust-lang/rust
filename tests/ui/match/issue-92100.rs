#![feature(half_open_range_patterns_in_slices)]

fn main() {
    match [1, 2] {
        [a.., a] => {} //~ ERROR cannot find value `a` in this scope
    }
}
