//@ known-bug: rust-lang/rust#129166

fn main() {
    #[cfg_eval]
    #[cfg]
    0
}
