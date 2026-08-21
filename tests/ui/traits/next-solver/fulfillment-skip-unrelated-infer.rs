//@ check-pass
//
// Typeck of a long chain of stalled `Default` obligations, then a
// constraint that resolves them. Next-solver fulfillment may skip
// walking the pending queue when only newer, unrelated infer vids
// changed (rustc#159933). This must still notice the final `u8`.

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
