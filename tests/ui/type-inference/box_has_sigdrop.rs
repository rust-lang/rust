//@ should-fail
//@ compile-flags: -Wrust-2021-incompatible-closure-captures
// Inference, canonicalization, and significant drops should work nicely together.
// Related issue: #86868

fn main() {
    let mut state = 0;
    Box::new(move || state)
}
