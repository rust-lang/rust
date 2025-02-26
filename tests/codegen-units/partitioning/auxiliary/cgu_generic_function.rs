#![crate_type = "lib"]

struct Struct(u32);

#[inline(never)]
pub fn foo<T>(x: T) -> (T, u32, i8) {
    let (x, Struct(y)) = bar(x);
    (x, y, 2)
}

#[inline(never)]
fn bar<T>(x: T) -> (T, Struct) {
    let _ = not_exported_and_not_generic(0);
    exported_and_generic::<u32>(0);
    (x, Struct(1))
}

pub static F: fn(u32) -> u32 = exported_and_generic::<u32>;

// These should not contribute to the codegen items of other crates.

// This is generic, but it's only instantiated with a u32 argument and that instantiation is present
// in the local crate (see F above).
#[inline(never)]
pub fn exported_and_generic<T>(x: T) -> T {
    x
}

#[inline(never)]
pub fn exported_but_not_generic(x: i32) -> i64 {
    x as i64
}

#[inline(never)]
fn not_exported_and_not_generic(x: u32) -> u64 {
    x as u64
}
