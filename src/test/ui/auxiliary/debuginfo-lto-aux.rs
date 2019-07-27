// compile-flags: -g --crate-type=rlib

pub struct StructWithLifetime<'a>(&'a i32);
pub fn mk_struct_with_lt<'a>(x: &'a i32) -> StructWithLifetime<'a> {
    StructWithLifetime(x)
}

pub struct RegularStruct(u32);
pub fn mk_regular_struct(x: u32) -> RegularStruct {
    RegularStruct(x)
}

pub fn take_fn(f: fn(i32) -> i32, x: i32) -> i32 {
    f(x)
}

pub fn with_closure(x: i32) -> i32 {
    let closure = |i| { x + i };

    closure(1) + closure(2)
}

pub fn generic_fn<T>(x: T) -> (T, u32) {
    (x, 1)
}

pub fn user_of_generic_fn(x: f32) -> (f32, u32) {
    generic_fn(x)
}
