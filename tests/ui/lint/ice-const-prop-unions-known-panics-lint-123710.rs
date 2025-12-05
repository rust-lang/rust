// Regression test for issue 123710.
// Tests that the we do not ICE in KnownPanicsLint
// when a union contains an enum with an repr(packed),
// which is a repr not supported for enums

#[repr(packed)]
//~^ ERROR attribute should be applied to a struct or union
#[repr(u32)]
enum E {
    A,
    B,
    C,
}

fn main() {
    union InvalidTag {
        int: u32,
        e: E,
//~^ ERROR field must implement `Copy` or be wrapped in `ManuallyDrop<...>` to be used in a union
    }
    let _invalid_tag = InvalidTag { int: 4 };
}
