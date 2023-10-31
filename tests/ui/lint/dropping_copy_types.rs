// check-pass

#![warn(dropping_copy_types)]

use std::mem::drop;
use std::vec::Vec;

#[derive(Copy, Clone)]
struct SomeStruct;

struct AnotherStruct {
    x: u8,
    y: u8,
    z: Vec<u8>,
}

impl Clone for AnotherStruct {
    fn clone(&self) -> AnotherStruct {
        AnotherStruct {
            x: self.x,
            y: self.y,
            z: self.z.clone(),
        }
    }
}

fn main() {
    let s1 = SomeStruct {};
    let s2 = s1;
    let s3 = &s1;
    let mut s4 = s1;
    let ref s5 = s1;

    drop(s1); //~ WARN calls to `std::mem::drop`
    drop(s2); //~ WARN calls to `std::mem::drop`
    drop(s3); //~ WARN calls to `std::mem::drop`
    drop(s4); //~ WARN calls to `std::mem::drop`
    drop(s5); //~ WARN calls to `std::mem::drop`

    let a1 = AnotherStruct {
        x: 255,
        y: 0,
        z: vec![1, 2, 3],
    };
    let a2 = &a1;
    let mut a3 = a1.clone();
    let ref a4 = a1;
    let a5 = a1.clone();

    drop(a2); //~ WARN calls to `std::mem::drop`
    drop(a3);
    drop(a4); //~ WARN calls to `std::mem::drop`
    drop(a5);
}

#[allow(unused)]
#[allow(clippy::unit_cmp)]
fn issue9482(x: u8) {
    fn println_and<T>(t: T) -> T {
        println!("foo");
        t
    }

    match x {
        // Don't lint (copy type), we only care about side-effects
        0 => drop(println_and(12)),
        // Don't lint (no copy type), we only care about side-effects
        1 => drop(println_and(String::new())),
        2 => {
            // Lint, even if we only care about the side-effect, it's already in a block
            drop(println_and(13)); //~ WARN calls to `std::mem::drop`
        },
         // Lint, idiomatic use is only in body of `Arm`
        3 if drop(println_and(14)) == () => (), //~ WARN calls to `std::mem::drop`
        // Lint, not a fn/method call
        4 => drop(2),//~ WARN calls to `std::mem::drop`
        _ => (),
    }
}

fn issue112653() {
    fn foo() -> Result<u8, ()> {
        println!("doing foo");
        Ok(0) // result is not always useful, the side-effect matters
    }
    fn bar() {
        println!("doing bar");
    }

    fn stuff() -> Result<(), ()> {
        match 42 {
            0 => drop(foo()?),  // drop is needed because we only care about side-effects
            1 => bar(),
            _ => (),  // doing nothing (no side-effects needed here)
        }
        Ok(())
    }
}
