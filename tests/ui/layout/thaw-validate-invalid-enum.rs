//@ compile-flags: -Zvalidate-mir

#[repr(packed)] //~ ERROR: attribute cannot be used on
#[repr(u32)]
enum E {
    A,
    B,
    C,
}

fn main() {
    union InvalidTag {
        int: u32,
        e: E, //~ ERROR: field must implement `Copy`
    }
    let _invalid_tag = InvalidTag { int: 4 };
}
