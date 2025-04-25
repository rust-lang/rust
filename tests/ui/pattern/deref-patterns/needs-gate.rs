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

    // `deref_patterns` allows string and byte string literals to have non-ref types.
    match *"test" {
        "test" => {}
        //~^ ERROR: mismatched types
        _ => {}
    }
    match *b"test" {
        b"test" => {}
        //~^ ERROR: mismatched types
        _ => {}
    }
    match *(b"test" as &[u8]) {
        b"test" => {}
        //~^ ERROR: mismatched types
        _ => {}
    }
}
