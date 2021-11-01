// FIXME: run-rustfix waiting on multi-span suggestions

#![warn(clippy::ref_binding_to_reference)]
#![allow(clippy::needless_borrowed_reference)]

fn f1(_: &str) {}
macro_rules! m2 {
    ($e:expr) => {
        f1(*$e)
    };
}
macro_rules! m3 {
    ($i:ident) => {
        Some(ref $i)
    };
}

#[allow(dead_code)]
fn main() {
    let x = String::new();

    // Ok, the pattern is from a macro
    let _: &&String = match Some(&x) {
        m3!(x) => x,
        None => return,
    };

    // Err, reference to a &String
    let _: &&String = match Some(&x) {
        Some(ref x) => x,
        None => return,
    };

    // Err, reference to a &String
    let _: &&String = match Some(&x) {
        Some(ref x) => {
            f1(x);
            f1(*x);
            x
        },
        None => return,
    };

    // Err, reference to a &String
    match Some(&x) {
        Some(ref x) => m2!(x),
        None => return,
    }

    // Err, reference to a &String
    let _ = |&ref x: &&String| {
        let _: &&String = x;
    };
}

// Err, reference to a &String
fn f2<'a>(&ref x: &&'a String) -> &'a String {
    let _: &&String = x;
    *x
}

trait T1 {
    // Err, reference to a &String
    fn f(&ref x: &&String) {
        let _: &&String = x;
    }
}

struct S;
impl T1 for S {
    // Err, reference to a &String
    fn f(&ref x: &&String) {
        let _: &&String = x;
    }
}
