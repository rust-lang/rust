#![allow(clippy::needless_borrow, clippy::useless_vec)]

#[deny(clippy::naive_bytecount)]
fn main() {
    let x = vec![0_u8; 16];

    let _ = x.iter().filter(|&&a| a == 0).count(); // naive byte count

    let _ = (&x[..]).iter().filter(|&a| *a == 0).count(); // naive byte count

    let _ = x.iter().filter(|a| **a > 0).count(); // not an equality count, OK.

    let _ = x.iter().map(|a| a + 1).filter(|&a| a < 15).count(); // not a slice

    let b = 0;

    let _ = x.iter().filter(|_| b > 0).count(); // woah there

    let _ = x.iter().filter(|_a| b == b + 1).count(); // nothing to see here, move along

    let _ = x.iter().filter(|a| b + 1 == **a).count(); // naive byte count

    let y = vec![0_u16; 3];

    let _ = y.iter().filter(|&&a| a == 0).count(); // naive count, but not bytes
}
