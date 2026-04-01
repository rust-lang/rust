//! Regression test for #139872
//! KnownPanicsLint used to assert ABI compatibility in the interpreter,
//! which ICEs with unsized statics.

enum E {
    V16(u16),
    V32(u32),
}

static C: (E, u16, str) = (E::V16(0xDEAD), 0x600D, 0xBAD);
//~^ ERROR the size for values of type `str` cannot be known

pub fn main() {
    let (_, n, _) = C;
}
