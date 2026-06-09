//@ run-pass
// Inference, canonicalization, and significant drops should work nicely together.
// Related issue: #86868

#[clippy::has_significant_drop]
struct DropGuy {}

fn creator() -> DropGuy {
    DropGuy {}
}

fn dropper() {
    let _ = creator();
}

fn main() {
    dropper();
}
