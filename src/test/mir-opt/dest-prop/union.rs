//! Tests that projections through unions cancel `DestinationPropagation`.

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
