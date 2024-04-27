#![crate_type = "lib"]

pub struct SomeType {
    pub some_member: usize,
}

pub static SOME_VALUE: SomeType = SomeType {
    some_member: 1,
};
