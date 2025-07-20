//@ known-bug: #139872

enum E {
    V16(u16),
    V32(u32),
}

static C: (E, u16, str) = (E::V16(0xDEAD), 0x600D, 0xBAD);

pub fn main() {
    let (_, n, _) = C;
}
