#![feature(plugin)]
#![plugin(clippy)]

fn main() {
    let x = vec![0_u8; 16];

    let _ = x.iter().filter(|&&a| a == 0).count(); // naive byte count

    let _ = (&x[..]).iter().filter(|&a| *a == 0).count(); // naive byte count

    let _ = x.iter().filter(|a| **a > 0).count(); // not an equality count, OK.

    let _ = x.iter().map(|a| a + 1).filter(|&a| a < 15).count(); // not a slice
}
