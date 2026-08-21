//@ check-pass
//
// Each `push(Default::default())` equates a fresh infer vid with the
// vec element vid. That is unknown-unknown unification, not instantiate.
// Next-solver fulfillment may skip the pending-queue walk for single-var
// stalls in that case (rustc#159933). Instantiating with `u8` must still
// resolve the stalled `Default` goals.

pub fn big() {
    let mut v = Vec::new();
    v.push(Default::default());
    v.push(Default::default());
    v.push(Default::default());
    v.push(Default::default());
    v.push(Default::default());
    v.push(Default::default());
    v.push(Default::default());
    v.push(Default::default());
    v.push(Default::default());
    v.push(Default::default());
    v.push(Default::default());
    v.push(Default::default());
    v.push(Default::default());
    v.push(Default::default());
    v.push(Default::default());
    v.push(Default::default());
    v.push(Default::default());
    v.push(Default::default());
    v.push(Default::default());
    v.push(Default::default());
    v.push(0u8);
}

fn main() {
    big();
}
