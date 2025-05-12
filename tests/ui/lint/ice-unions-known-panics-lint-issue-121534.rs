// Regression test for #121534
// Tests that no ICE occurs in KnownPanicsLint when it
// evaluates an operation whose operands have different
// layout types even though they have the same type.
// This situation can be contrived through the use of
// unions as in this test

//@ build-pass
union Union {
    u32_field: u32,
    i32_field: i32,
}

pub fn main() {
    let u32_variant = Union { u32_field: 2 };
    let i32_variant = Union { i32_field: 3 };
    let a = unsafe { u32_variant.u32_field };
    let b = unsafe { i32_variant.u32_field };

    let _diff = a - b;
}
