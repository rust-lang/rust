// compile-flags: -Z parse-only

fn that_odd_parse() {
    // see assoc-oddities-1 for explanation
    x..if c { a } else { b }[n]; //~ ERROR expected one of
}
