//@ check-pass

// Test `@` patterns combined with `box` patterns.

#![allow(dropping_references)]
#![allow(dropping_copy_types)]

#![feature(box_patterns)]

#[derive(Copy, Clone)]
struct C;

fn c() -> C { C }

struct NC;

fn nc() -> NC { NC }

fn main() {
    let ref a @ box b = Box::new(C); // OK; the type is `Copy`.
    drop(b);
    drop(b);
    drop(a);

    let ref a @ box b = Box::new(c()); // OK; the type is `Copy`.
    drop(b);
    drop(b);
    drop(a);

    fn f3(ref a @ box b: Box<C>) { // OK; the type is `Copy`.
        drop(b);
        drop(b);
        drop(a);
    }
    match Box::new(c()) {
        ref a @ box b => { // OK; the type is `Copy`.
            drop(b);
            drop(b);
            drop(a);
        }
    }

    let ref a @ box ref b = Box::new(NC); // OK.
    drop(a);
    drop(b);

    fn f4(ref a @ box ref b: Box<NC>) { // OK.
        drop(a);
        drop(b)
    }

    match Box::new(nc()) {
        ref a @ box ref b => { // OK.
            drop(a);
            drop(b);
        }
    }

    match Box::new([Ok(c()), Err(nc()), Ok(c())]) {
        box [Ok(a), ref xs @ .., Err(ref b)] => {
            let _: C = a;
            let _: &[Result<C, NC>; 1] = xs;
            let _: &NC = b;
        }
        _ => {}
    }

    match [Ok(Box::new(c())), Err(Box::new(nc())), Ok(Box::new(c())), Ok(Box::new(c()))] {
        [Ok(box a), ref xs @ .., Err(box ref b), Err(box ref c)] => {
            let _: C = a;
            let _: &[Result<Box<C>, Box<NC>>; 1] = xs;
            let _: &NC = b;
            let _: &NC = c;
        }
        _ => {}
    }

    match Box::new([Ok(c()), Err(nc()), Ok(c())]) {
        box [Ok(a), ref xs @ .., Err(b)] => {}
        _ => {}
    }

    match [Ok(Box::new(c())), Err(Box::new(nc())), Ok(Box::new(c())), Ok(Box::new(c()))] {
        [Ok(box ref a), ref xs @ .., Err(box b), Err(box ref mut c)] => {}
        _ => {}
    }
}
