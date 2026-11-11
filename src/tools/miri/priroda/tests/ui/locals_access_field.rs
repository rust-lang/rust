// Verifies that the `locals` command lists locals introduced by accessing
// struct fields, including the original aggregate and each extracted field.
struct ExtraSlice<'a> {
    _slice: &'a [u8],
    _extra: u32,
}

impl<'a> ExtraSlice<'a> {
    fn new() -> ExtraSlice<'a> {
        ExtraSlice { _slice: &[1; 0], _extra: 0 }
    }
}

fn main() {
    let extraslice = ExtraSlice::new();
    let _slice = extraslice._slice;
    let _extra = extraslice._extra;
}
