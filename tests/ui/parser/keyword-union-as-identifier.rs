// check-pass

#![allow(non_camel_case_types)]

mod union {
    type union = i32;

    pub struct Bar {
        pub union: union,
    }

    pub fn union() -> Bar {
        Bar {
            union: 5
        }
    }
}

fn main() {
    let union = union::union();
    let _ = union.union;
}
