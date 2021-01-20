#![crate_type = "lib"]

#[repr(C)]
pub struct TestUnion {
    _val: u64
}

#[link(name = "ctest", kind = "static")]
extern {
    pub fn give_back(tu: TestUnion) -> u64;
}
