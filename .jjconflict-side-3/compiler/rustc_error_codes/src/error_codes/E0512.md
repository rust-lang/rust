Transmute with two differently sized types was attempted.

Erroneous code example:

```compile_fail,E0512
fn takes_u8(_: u8) {}

fn main() {
    unsafe { takes_u8(::std::mem::transmute(0u16)); }
    // error: cannot transmute between types of different sizes,
    //        or dependently-sized types
}
```

Please use types with same size or use the expected type directly. Example:

```
fn takes_u8(_: u8) {}

fn main() {
    unsafe { takes_u8(::std::mem::transmute(0i8)); } // ok!
    // or:
    unsafe { takes_u8(0u8); } // ok!
}
```
