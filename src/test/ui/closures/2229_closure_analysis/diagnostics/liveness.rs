// edition:2021

// check-pass
#![allow(unreachable_code)]
#![warn(unused)]
#![allow(dead_code)]

#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
}

pub fn f() {
    let mut a = 1;
    let mut c = Point{ x:1, y:0 };

    // Captured by value, but variable is dead on entry.
    (move || {
        // This will not trigger a warning for unused variable as
        // c.x will be treated as a Non-tracked place
        c.x = 1;
        println!("{}", c.x);
        a = 1; //~ WARN value captured by `a` is never read
        println!("{}", a);
    })();

    // Read and written to, but never actually used.
    (move || {
        // This will not trigger a warning for unused variable as
        // c.x will be treated as a Non-tracked place
        c.x += 1;
        a += 1; //~ WARN unused variable: `a`
    })();

    (move || {
        println!("{}", c.x);
        // Value is read by closure itself on later invocations.
        // This will not trigger a warning for unused variable as
        // c.x will be treated as a Non-tracked place
        c.x += 1;
        println!("{}", a);
        a += 1;
    })();
    let b = Box::new(42);
    (move || {
        println!("{}", c.x);
        // Never read because this is FnOnce closure.
        // This will not trigger a warning for unused variable as
        // c.x will be treated as a Non-tracked place
        c.x += 1;
        println!("{}", a);
        a += 1; //~ WARN value assigned to `a` is never read
        drop(b);
    })();
}

#[derive(Debug)]
struct MyStruct<'a>  {
    x: Option<& 'a str>,
    y: i32,
}

pub fn nested() {
    let mut a : Option<& str>;
    a = None;
    let mut b : Option<& str>;
    b = None;
    let mut d = MyStruct{ x: None, y: 1};
    let mut e = MyStruct{ x: None, y: 1};
    (|| {
        (|| {
            // This will not trigger a warning for unused variable as
            // d.x will be treated as a Non-tracked place
            d.x = Some("d1");
            d.x = Some("d2");
            a = Some("d1"); //~ WARN value assigned to `a` is never read
            a = Some("d2");
        })();
        (move || {
            // This will not trigger a warning for unused variable as
            //e.x will be treated as a Non-tracked place
            e.x = Some("e1");
            e.x = Some("e2");
            b = Some("e1"); //~ WARN value assigned to `b` is never read
                            //~| WARN unused variable: `b`
            b = Some("e2"); //~ WARN value assigned to `b` is never read
        })();
    })();
}

fn main() {}
