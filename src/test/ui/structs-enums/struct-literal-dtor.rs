// run-pass
#![allow(non_camel_case_types)]

struct foo {
    x: String,
}

impl Drop for foo {
    fn drop(&mut self) {
        println!("{}", self.x);
    }
}

pub fn main() {
    let _z = foo {
        x: "Hello".to_string()
    };
}
