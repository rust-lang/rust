// run-pass
// pretty-expanded FIXME #23616

#[derive(Clone)]
struct S {
    _int: isize,
    _i8: i8,
    _i16: i16,
    _i32: i32,
    _i64: i64,

    _uint: usize,
    _u8: u8,
    _u16: u16,
    _u32: u32,
    _u64: u64,

    _f32: f32,
    _f64: f64,

    _bool: bool,
    _char: char,
    _nil: ()
}

pub fn main() {}
