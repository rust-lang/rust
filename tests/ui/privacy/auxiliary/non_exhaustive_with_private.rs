// Auxiliary crate for testing non-exhaustive struct with private fields

#[non_exhaustive]
pub struct Foo {
    pub my_field: u32,
    private_field: i32,
}

#[non_exhaustive]
pub struct Bar {
    pub my_field: u32,
    pub missing_field: i32,
}
