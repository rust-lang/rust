// run-pass
#![feature(let_else)]

use std::fmt::Display;
use std::rc::Rc;
use std::sync::atomic::{AtomicU8, Ordering};

static TRACKER: AtomicU8 = AtomicU8::new(0);

#[derive(Default)]
struct Droppy {
    inner: u32,
}

impl Drop for Droppy {
    fn drop(&mut self) {
        TRACKER.store(1, Ordering::Release);
        println!("I've been dropped");
    }
}

fn foo<'a>(x: &'a str) -> Result<impl Display + 'a, ()> {
    Ok(x)
}

fn main() {
    assert_eq!(TRACKER.load(Ordering::Acquire), 0);
    let 0 = Droppy::default().inner else { return };
    assert_eq!(TRACKER.load(Ordering::Acquire), 1);
    println!("Should have dropped 👆");

    {
        let x = String::from("Hey");

        let Ok(s) = foo(&x) else { panic!() };
        assert_eq!(s.to_string(), x);
    }
    {
        // test let-else drops temps after statement
        let rc = Rc::new(0);
        let 0 = *rc.clone() else { unreachable!() };
        Rc::try_unwrap(rc).unwrap();
    }
    {
        let mut rc = Rc::new(0);
        let mut i = 0;
        loop {
            if i > 3 {
                break;
            }
            let 1 = *rc.clone() else {
                if let Ok(v) = Rc::try_unwrap(rc) {
                    rc = Rc::new(v);
                } else {
                    panic!()
                }
                i += 1;
                continue
            };
        }
    }
    {
        // test let-else drops temps before else block
        let rc = Rc::new(0);
        let 1 = *rc.clone() else {
            Rc::try_unwrap(rc).unwrap();
            return;
        };
        unreachable!();
    }
}
