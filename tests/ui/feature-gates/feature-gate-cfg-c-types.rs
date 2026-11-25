pub type Char = cfg_select! {
    c_char = "i8" => i8, //~ ERROR `cfg(c_char)` is experimental
    c_char = "u8" => u8, //~ ERROR `cfg(c_char)` is experimental
};

pub type SChar = cfg_select! {
    c_schar = "i8" => i8, //~ ERROR `cfg(c_schar)` is experimental
};

pub type UChar = cfg_select! {
    c_uchar = "u8" => u8, //~ ERROR `cfg(c_uchar)` is experimental
};

pub type Short = cfg_select! {
    c_short = "i16" => i16, //~ ERROR `cfg(c_short)` is experimental
};

pub type UShort = cfg_select! {
    c_ushort = "u16" => u16, //~ ERROR `cfg(c_ushort)` is experimental
};

pub type Int = cfg_select! {
    c_int = "i16" => i16, //~ ERROR `cfg(c_int)` is experimental
    c_int = "i32" => i32, //~ ERROR `cfg(c_int)` is experimental
};

pub type UInt = cfg_select! {
    c_uint = "u16" => u16, //~ ERROR `cfg(c_uint)` is experimental
    c_uint = "u32" => u32, //~ ERROR `cfg(c_uint)` is experimental
};

pub type Long = cfg_select! {
    c_long = "i32" => i32, //~ ERROR `cfg(c_long)` is experimental
    c_long = "i64" => i64, //~ ERROR `cfg(c_long)` is experimental
};

pub type ULong = cfg_select! {
    c_ulong = "u32" => u32, //~ ERROR `cfg(c_ulong)` is experimental
    c_ulong = "u64" => u64, //~ ERROR `cfg(c_ulong)` is experimental
};

pub type LongLong = cfg_select! {
    c_longlong = "i64" => i64, //~ ERROR `cfg(c_longlong)` is experimental
};

pub type ULongLong = cfg_select! {
    c_ulonglong = "u64" => u64, //~ ERROR `cfg(c_ulonglong)` is experimental
};

pub type Float = cfg_select! {
    c_float = "f32" => f32, //~ ERROR `cfg(c_float)` is experimental
};

pub type Double = cfg_select! {
    c_double = "f32" => f32, //~ ERROR `cfg(c_double)` is experimental
    c_double = "f64" => f64, //~ ERROR `cfg(c_double)` is experimental
};

fn main() {}
