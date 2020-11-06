// ignore-wasm32-bare compiled with panic=abort by default
#![crate_type = "lib"]

// EMIT_MIR inline_shims.clone.Inline.diff
pub fn clone<A, B>(f: fn(A, B)) -> fn(A, B) {
    f.clone()
}

// EMIT_MIR inline_shims.drop.Inline.diff
pub fn drop<A, B>(a: *mut Vec<A>, b: *mut Option<B>) {
    unsafe { std::ptr::drop_in_place(a) }
    unsafe { std::ptr::drop_in_place(b) }
}
