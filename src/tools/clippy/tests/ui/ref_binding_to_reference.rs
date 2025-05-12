// FIXME: run-rustfix waiting on multi-span suggestions
//@no-rustfix
#![warn(clippy::ref_binding_to_reference)]
#![allow(clippy::needless_borrowed_reference, clippy::explicit_auto_deref)]

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
        //~^ ref_binding_to_reference
        None => return,
    };

    // Err, reference to a &String
    let _: &&String = match Some(&x) {
        Some(ref x) => {
            //~^ ref_binding_to_reference

            f1(x);
            f1(*x);
            x
        },
        None => return,
    };

    // Err, reference to a &String
    match Some(&x) {
        Some(ref x) => m2!(x),
        //~^ ref_binding_to_reference
        None => return,
    }

    // Err, reference to a &String
    let _ = |&ref x: &&String| {
        //~^ ref_binding_to_reference

        let _: &&String = x;
    };
}

// Err, reference to a &String
fn f2<'a>(&ref x: &&'a String) -> &'a String {
    //~^ ref_binding_to_reference

    let _: &&String = x;
    *x
}

trait T1 {
    // Err, reference to a &String
    fn f(&ref x: &&String) {
        //~^ ref_binding_to_reference

        let _: &&String = x;
    }
}

struct S;
impl T1 for S {
    // Err, reference to a &String
    fn f(&ref x: &&String) {
        //~^ ref_binding_to_reference

        let _: &&String = x;
    }
}

fn check_expect_suppression() {
    let x = String::new();
    #[expect(clippy::ref_binding_to_reference)]
    let _: &&String = match Some(&x) {
        Some(ref x) => x,
        None => return,
    };
}
