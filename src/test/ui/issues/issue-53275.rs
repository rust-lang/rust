// run-pass

#![crate_type = "lib"]
struct S(u8);

pub fn ice() {
    S([][0]);
}
