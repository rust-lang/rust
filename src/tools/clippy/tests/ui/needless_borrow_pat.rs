// FIXME: run-rustfix waiting on multi-span suggestions

#![warn(clippy::needless_borrow)]
#![allow(clippy::needless_borrowed_reference, clippy::explicit_auto_deref)]

fn f1(_: &str) {}
macro_rules! m1 {
    ($e:expr) => {
        f1($e)
    };
}
macro_rules! m3 {
    ($i:ident) => {
        Some(ref $i)
    };
}
macro_rules! if_chain {
    (if $e:expr; $($rest:tt)*) => {
        if $e {
            if_chain!($($rest)*)
        }
    };

    (if let $p:pat = $e:expr; $($rest:tt)*) => {
        if let $p = $e {
            if_chain!($($rest)*)
        }
    };

    (then $b:block) => {
        $b
    };
}

#[allow(dead_code)]
fn main() {
    let x = String::new();

    // Ok, reference to a String.
    let _: &String = match Some(x.clone()) {
        Some(ref x) => x,
        None => return,
    };

    // Ok, reference to a &mut String
    let _: &&mut String = match Some(&mut x.clone()) {
        Some(ref x) => x,
        None => return,
    };

    // Ok, the pattern is from a macro
    let _: &String = match Some(&x) {
        m3!(x) => x,
        None => return,
    };

    // Err, reference to a &String
    let _: &String = match Some(&x) {
        Some(ref x) => x,
        //~^ needless_borrow
        None => return,
    };

    // Err, reference to a &String.
    let _: &String = match Some(&x) {
        Some(ref x) => *x,
        //~^ needless_borrow
        None => return,
    };

    // Err, reference to a &String
    let _: &String = match Some(&x) {
        Some(ref x) => {
            //~^ needless_borrow

            f1(x);
            f1(*x);
            x
        },
        None => return,
    };

    // Err, reference to a &String
    match Some(&x) {
        Some(ref x) => m1!(x),
        //~^ needless_borrow
        None => return,
    };

    // Err, reference to a &String
    let _ = |&ref x: &&String| {
        //~^ needless_borrow

        let _: &String = x;
    };

    // Err, reference to a &String
    let (ref y,) = (&x,);
    //~^ needless_borrow

    let _: &String = *y;

    let y = &&x;
    // Ok, different y
    let _: &String = *y;

    let x = (0, 0);
    // Err, reference to a &u32. Don't suggest adding a reference to the field access.
    let _: u32 = match Some(&x) {
        Some(ref x) => x.0,
        //~^ needless_borrow
        None => return,
    };

    enum E {
        A(&'static u32),
        B(&'static u32),
    }
    // Err, reference to &u32.
    let _: &u32 = match E::A(&0) {
        E::A(ref x) | E::B(ref x) => *x,
        //~^ needless_borrow
    };

    // Err, reference to &String.
    if_chain! {
        if true;
        if let Some(ref x) = Some(&String::new());
        //~^ needless_borrow

        then {
            f1(x);
        }
    }
}

// Err, reference to a &String
fn f2<'a>(&ref x: &&'a String) -> &'a String {
    //~^ needless_borrow

    let _: &String = x;
    *x
}

trait T1 {
    // Err, reference to a &String
    fn f(&ref x: &&String) {
        //~^ needless_borrow

        let _: &String = x;
    }
}

struct S;
impl T1 for S {
    // Err, reference to a &String
    fn f(&ref x: &&String) {
        //~^ needless_borrow

        let _: &String = *x;
    }
}

// Ok - used to error due to rustc bug
#[allow(dead_code)]
#[derive(Debug)]
enum Foo<'a> {
    Str(&'a str),
}
