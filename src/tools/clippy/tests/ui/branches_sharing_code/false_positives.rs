#![allow(dead_code)]
#![deny(clippy::if_same_then_else, clippy::branches_sharing_code)]

// ##################################
// # Issue clippy#7369
// ##################################
#[derive(Debug)]
pub struct FooBar {
    foo: Vec<u32>,
}

impl FooBar {
    pub fn bar(&mut self) {
        if true {
            self.foo.pop();
        } else {
            self.baz();

            self.foo.pop();

            self.baz()
        }
    }

    fn baz(&mut self) {}
}

fn foo(x: u32, y: u32) -> u32 {
    x / y
}

fn main() {
    let x = (1, 2);
    let _ = if true {
        let (x, y) = x;
        foo(x, y)
    } else {
        let (y, x) = x;
        foo(x, y)
    };
}
