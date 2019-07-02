// build-pass (FIXME(62277): could be check-pass?)
#![allow(dead_code)]
pub struct Struct<K: 'static> {
    pub field: K,
}

static STRUCT: Struct<&'static [u8]> = Struct {
    field: {&[1]}
};

static STRUCT2: Struct<&'static [u8]> = Struct {
    field: &[1]
};

fn main() {}
