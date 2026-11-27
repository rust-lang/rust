// Regression test for <https://github.com/rust-lang/rust/issues/138069>
// This test ensures that unused_assignments does not report assignments used in a match.
//@ check-pass

fn pnk(x: usize) -> &'static str {
    let mut k1 = "k1";
    let mut h1 = "h1";
    match x & 3 {
        3 if { k1 = "unused?"; false } => (),
        _ if { h1 = k1; true } => (),
        _ => (),
    }
    h1
}

#[deny(unused_assignments)]
fn main() {
    pnk(3);
}
