// compile-flags: -Z parse-only

fn main() {
    // following lines below parse and must not fail
    x = if c { a } else { b }();
    x = if true { 1 } else { 0 } as *mut _;
    // however this does not parse and probably should fail to retain compat?
    // N.B., `..` here is arbitrary, failure happens/should happen ∀ops that aren’t `=`
    // see assoc-oddities-2 and assoc-oddities-3
    ..if c { a } else { b }[n]; //~ ERROR expected one of
}
