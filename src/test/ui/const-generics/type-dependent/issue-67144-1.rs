// check-pass
// revisions: full min
#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]

struct X;

impl X {
    pub fn getn<const N: usize>(&self) -> [u8; N] {
        getn::<N>()
    }
}

fn getn<const N: usize>() -> [u8; N] {
    unsafe {
        std::mem::zeroed()
    }
}

fn main() {
    // works
    let [a,b,c] = getn::<3>();

    // cannot pattern-match on an array without a fixed length
    let [a,b,c] = X.getn::<3>();

    // mismatched types, expected array `[u8; 3]` found array `[u8; _]`
    let arr: [u8; 3] = X.getn::<3>();
}
