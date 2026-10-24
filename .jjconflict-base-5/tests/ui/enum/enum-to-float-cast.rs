// Tests that enum-to-float casts are disallowed.

enum E {
    L0 = -1,
    H0 = 1
}

enum F {
    L1 = 1,
    H1 = 0xFFFFFFFFFFFFFFFF
}

static C0: f32 = E::L0 as f32; //~ ERROR casting
static C1: f32 = F::H1 as f32; //~ ERROR casting

pub fn main() {
    let b = C0;
    let d = C1;
    assert_eq!(b, -1.0f32);
    assert_eq!(d, -1.0f32);
}
