//@ check-pass

#![allow(dead_code)]
#![deny(clippy::if_same_then_else, clippy::branches_sharing_code)]

use std::sync::Mutex;

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

    let m = Mutex::new(0u32);
    let l = m.lock().unwrap();
    let _ = if true {
        drop(l);
        println!("foo");
        m.lock().unwrap();
        0
    } else if *l == 0 {
        drop(l);
        println!("foo");
        println!("bar");
        m.lock().unwrap();
        1
    } else {
        drop(l);
        println!("foo");
        println!("baz");
        m.lock().unwrap();
        2
    };

    if true {
        let _guard = m.lock();
        println!("foo");
    } else {
        println!("foo");
    }

    if true {
        let _guard = m.lock();
        println!("foo");
        println!("bar");
    } else {
        let _guard = m.lock();
        println!("foo");
        println!("baz");
    }

    let mut c = 0;
    for _ in 0..5 {
        if c == 0 {
            c += 1;
            println!("0");
        } else if c == 1 {
            c += 1;
            println!("1");
        } else {
            c += 1;
            println!("more");
        }
    }
}
