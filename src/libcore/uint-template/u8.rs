type T = u8;
const bits: uint = 8;

// Type-specific functions here. These must be reexported by the
// parent module so that they appear in core::u8 and not core::u8::u8;

pure fn is_ascii(x: T) -> bool { return 0 as T == x & 128 as T; }
