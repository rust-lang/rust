//! Test that we allow unsizing `[Opaque; N]` to `[Concrete]`.
#![allow(todo_macro_calls)]

//@check-pass

fn hello() -> [impl Sized; 2] {
    if false {
        let x = hello();
        let _: &[i32] = &x;
    }
    todo!()
}

fn main() {}
