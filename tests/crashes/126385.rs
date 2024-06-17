//@ known-bug: rust-lang/rust#126385
pub struct MyStruct<'field> {
    field: &'_ [u32],
}

impl MyStruct<'_> {
    pub fn _<'a>(field: &'a[u32]) -> Self<new> {
    Self{field}
    }
}
