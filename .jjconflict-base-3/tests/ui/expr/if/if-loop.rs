//@ check-pass

// This used to ICE because the "if" being unreachable was not handled correctly
fn err() {
    if loop {} {}
}

fn main() {}
