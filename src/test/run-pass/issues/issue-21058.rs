// run-pass
#![allow(dead_code)]
#![feature(core_intrinsics)]

struct NT(str);
struct DST { a: u32, b: str }

fn main() {
    // type_name should support unsized types
    assert_eq!(unsafe {(
        // Slice
        std::intrinsics::type_name::<[u8]>(),
        // str
        std::intrinsics::type_name::<str>(),
        // Trait
        std::intrinsics::type_name::<Send>(),
        // Newtype
        std::intrinsics::type_name::<NT>(),
        // DST
        std::intrinsics::type_name::<DST>()
    )}, ("[u8]", "str", "dyn std::marker::Send", "NT", "DST"));
}
