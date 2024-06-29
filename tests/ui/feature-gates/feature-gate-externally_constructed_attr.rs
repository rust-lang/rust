#[repr(C)]
#[externally_constructed] //~ ERROR the `#[mark_externally_constructed]` attribute is an experimental feature
pub struct Foo {
    pub i: i16,
    align: i16
}

fn main() {}
