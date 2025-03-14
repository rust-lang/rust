// gate-test-deref_patterns

fn main() {
    match Box::new(0) {
        deref!(0) => {}
        //~^ ERROR: use of unstable library feature `deref_patterns`: placeholder syntax for deref patterns
        _ => {}
    }

    match Box::new(0) {
        0 => {}
        //~^ ERROR: mismatched types
        _ => {}
    }
}
