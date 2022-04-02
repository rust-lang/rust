//! Tests that we can propogate into places that are projections into unions
// compile-flags: -Zunsound-mir-opts
fn val() -> u32 {
    1
}

// EMIT_MIR union.main.DestinationPropagation.diff
fn main() {
    union Un {
        us: u32,
    }

    let un = Un { us: val() };

    drop(unsafe { un.us });
}

// FIXME(JakobDegen): This example is currently broken; it needs a more precise liveness analysis in
// order to be fixed.
