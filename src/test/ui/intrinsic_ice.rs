// Test for https://github.com/rust-lang/rust/issues/34123
// ICE: intrinsic .. being reified

#![feature(intrinsics)]

fn main(){
    let transmute = std::intrinsics::transmute;
    let assign: unsafe extern "rust-intrinsic" fn(*const i32) -> *mut i32 = transmute;
}
