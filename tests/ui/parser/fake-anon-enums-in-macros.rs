// build-pass
macro_rules! check_ty {
    ($Z:ty) => { compile_error!("triggered"); };
    ($X:ty | $Y:ty) => { $X };
}

macro_rules! check {
    ($Z:ty) => { compile_error!("triggered"); };
    ($X:ty | $Y:ty) => { };
}

check! { i32 | u8 }

fn foo(x: check_ty! { i32 | u8 }) -> check_ty! { i32 | u8 } {
    x
}
fn main() {
    let x: check_ty! { i32 | u8 } = 42;
    let _: check_ty! { i32 | u8 } = foo(x);
}
