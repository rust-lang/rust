//@ compile-flags: -Zmir-opt-level=0 -Zmir-enable-passes=+ScalarReplacementOfAggregates

// Source only declares `s` and `_slice` in `extra`. But after the
// ScalarReplacementOfAggregates optimization, rustc splits the struct into fields. Priroda should
// show those split field locals as separate MIR locals with the projected source debug paths.

pub struct ExtraSlice<'a> {
    _slice: &'a [u8],
    _extra: u32,
}

pub fn extra(s: &[u8]) {
    let _slice = ExtraSlice { _slice: s, _extra: s.len() as u32 };
}

fn main() {
    let bytes = &[3, 3];
    extra(bytes);
}
