#![crate_name = "foo"]

pub struct Struc;

//@ has foo/struct.Struc.html
//@ has - '//*[@id="main-content"]/h2[@id="implementations"]' "Implementations"
impl Struc {
    pub const S: u64 = 0;
}
