//@ check-pass

#[allow(clippy::no_effect)]
fn main() {
    #[clippy::author]
    [1_u8; 5];
}
