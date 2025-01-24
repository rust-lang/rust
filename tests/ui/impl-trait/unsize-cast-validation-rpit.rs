//@ check-pass
//@ compile-flags: -Zvalidate-mir

fn hello() -> &'static [impl Sized; 0] {
    if false {
        let x = hello();
        let _: &[i32] = x;
    }
    &[]
}

fn main() {}
