//@ known-bug: rust-lang/rust#126359

struct OppOrder<const N: u8 = 3, T = u32> {
    arr: [T; N],
}

fn main() {
    let _ = OppOrder::<3, u32> { arr: [0, 0, 0] };
}
