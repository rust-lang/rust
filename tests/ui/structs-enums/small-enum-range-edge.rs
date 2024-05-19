//@ run-pass
#![allow(non_upper_case_globals)]

// Tests the range assertion wraparound case when reading discriminants.

#[repr(u8)]
#[derive(Copy, Clone)]
enum Eu { Lu = 0, Hu = 255 }

static CLu: Eu = Eu::Lu;
static CHu: Eu = Eu::Hu;

#[repr(i8)]
#[derive(Copy, Clone)]
enum Es { Ls = -128, Hs = 127 }

static CLs: Es = Es::Ls;
static CHs: Es = Es::Hs;

pub fn main() {
    assert_eq!((Eu::Hu as u8).wrapping_add(1), Eu::Lu as u8);
    assert_eq!((Es::Hs as i8).wrapping_add(1), Es::Ls as i8);
    assert_eq!(CLu as u8, Eu::Lu as u8);
    assert_eq!(CHu as u8, Eu::Hu as u8);
    assert_eq!(CLs as i8, Es::Ls as i8);
    assert_eq!(CHs as i8, Es::Hs as i8);
}
