//@ compile-flags: -Z mir-opt-level=3
//@ build-pass
#![crate_type="lib"]

// This used to ICE: const-prop did not account for field reordering of scalar pairs,
// and would generate a tuple like `(0x1337, VariantBar): (FooEnum, isize)`,
// causing assertion failures in codegen when trying to read 0x1337 at the wrong type.

pub enum FooEnum {
    VariantBar,
    VariantBaz,
    VariantBuz,
}

pub fn wrong_index() -> isize {
    let (_, b) = id((FooEnum::VariantBar, 0x1337));
    b
}

pub fn wrong_index_two() -> isize {
    let (_, (_, b)) = id(((), (FooEnum::VariantBar, 0x1338)));
    b
}

fn id<T>(x: T) -> T {
    x
}
