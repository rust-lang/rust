//@ known-bug: rust-lang/rust#124583

fn main() {
    let _ = -(-0.0f16);
}
