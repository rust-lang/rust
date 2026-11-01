#![crate_type="lib"]

pub struct S {
    x: isize,
}

impl Drop for S {
    fn drop(&mut self) {
        println!("goodbye");
    }
}

pub fn f() {
    let x = S { x: 1 };
    let y = x;
    let _z = y;
}
