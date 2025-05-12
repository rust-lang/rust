//@ compile-flags: -Z parse-crate-root-only

fn main() {
    // see assoc-oddities-1 for explanation
    x..if c { a } else { b }[n]; //~ ERROR expected one of
}
