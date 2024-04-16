//@ known-bug: #123710

#[repr(packed)]
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
    }
    let _invalid_tag = InvalidTag { int: 4 };
}
