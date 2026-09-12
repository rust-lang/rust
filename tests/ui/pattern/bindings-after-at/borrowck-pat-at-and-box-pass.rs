//@ check-pass

// Test `@` patterns combined with `deref!` patterns.

#![allow(dropping_references)]
#![allow(dropping_copy_types)]

#![feature(deref_patterns)]

#[derive(Copy, Clone)]
struct C;

fn c() -> C { C }

struct NC;

fn nc() -> NC { NC }

fn main() {
    let ref a @ deref!(b) = Box::new(C); // OK; the type is `Copy`.
    drop(b);
    drop(b);
    drop(a);

    let ref a @ deref!(b) = Box::new(c()); // OK; the type is `Copy`.
    drop(b);
    drop(b);
    drop(a);

    fn f3(ref a @ deref!(b): Box<C>) { // OK; the type is `Copy`.
        drop(b);
        drop(b);
        drop(a);
    }
    match Box::new(c()) {
        ref a @ deref!(b) => { // OK; the type is `Copy`.
            drop(b);
            drop(b);
            drop(a);
        }
    }

    let ref a @ deref!(ref b) = Box::new(NC); // OK.
    drop(a);
    drop(b);

    fn f4(ref a @ deref!(ref b): Box<NC>) { // OK.
        drop(a);
        drop(b)
    }

    match Box::new(nc()) {
        ref a @ deref!(ref b) => { // OK.
            drop(a);
            drop(b);
        }
    }

    match Box::new([Ok(c()), Err(nc()), Ok(c())]) {
        deref!([Ok(a), ref xs @ .., Err(ref b)]) => {
            let _: C = a;
            let _: &[Result<C, NC>; 1] = xs;
            let _: &NC = b;
        }
        _ => {}
    }

    match [Ok(Box::new(c())), Err(Box::new(nc())), Ok(Box::new(c())), Ok(Box::new(c()))] {
        [Ok(deref!(a)), ref xs @ .., Err(deref!(ref b)), Err(deref!(ref c))] => {
            let _: C = a;
            let _: &[Result<Box<C>, Box<NC>>; 1] = xs;
            let _: &NC = b;
            let _: &NC = c;
        }
        _ => {}
    }

    match Box::new([Ok(c()), Err(nc()), Ok(c())]) {
        deref!([Ok(a), ref xs @ .., Err(b)]) => {}
        _ => {}
    }

    match [Ok(Box::new(c())), Err(Box::new(nc())), Ok(Box::new(c())), Ok(Box::new(c()))] {
        [Ok(deref!(ref a)), ref xs @ .., Err(deref!(b)), Err(deref!(ref mut c))] => {}
        _ => {}
    }
}
