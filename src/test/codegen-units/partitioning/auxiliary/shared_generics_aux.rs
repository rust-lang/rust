// compile-flags:-Zshare-generics=yes

#![crate_type="rlib"]

pub fn generic_fn<T>(x: T, y: T) -> (T, T) {
    (x, y)
}

pub fn use_generic_fn_f32() -> (f32, f32) {
    generic_fn(0.0f32, 1.0f32)
}
