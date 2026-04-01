//! Test that we allow unsizing `[Opaque; N]` to `[Concrete]`.

//@check-pass

fn hello() -> [impl Sized; 2] {
    if false {
        let x = hello();
        let _: &[i32] = &x;
    }
    todo!()
}

fn main() {}
