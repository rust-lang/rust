//! Test that coercing between function items of the same function,
//! but with different generic args succeeds in typeck, but then fails
//! in borrowck when the lifetimes can't actually be merged.

fn foo<T>(t: T) -> T {
    t
}

fn f<'a, 'b, 'c: 'a + 'b>(a: &'a (), b: &'b (), c: &'c ()) {
    let mut x = foo::<&'a ()>; //~ ERROR: lifetime may not live long enough
    x = foo::<&'b ()>; //~ ERROR: lifetime may not live long enough
    x = foo::<&'c ()>;
    x(a);
    x(b);
    x(c);
}

fn g<'a, 'b, 'c: 'a + 'b>(a: &'a (), b: &'b (), c: &'c ()) {
    let x = foo::<&'c ()>;
    let _: &'c () = x(a); //~ ERROR lifetime may not live long enough
}

fn h<'a, 'b, 'c: 'a + 'b>(a: &'a (), b: &'b (), c: &'c ()) {
    let x = foo::<&'a ()>;
    let _: &'a () = x(c);
}

fn i<'a, 'b, 'c: 'a + 'b>(a: &'a (), b: &'b (), c: &'c ()) {
    let mut x = foo::<&'c ()>;
    x = foo::<&'b ()>; //~ ERROR lifetime may not live long enough
    x = foo::<&'a ()>; //~ ERROR lifetime may not live long enough
    x(a);
    x(b);
    x(c);
}

fn j<'a, 'b, 'c: 'a + 'b>(a: &'a (), b: &'b (), c: &'c ()) {
    let x = match true {
        true => foo::<&'b ()>,  //~ ERROR lifetime may not live long enough
        false => foo::<&'a ()>, //~ ERROR lifetime may not live long enough
    };
    x(a);
    x(b);
    x(c);
}

fn k<'a, 'b, 'c: 'a + 'b>(a: &'a (), b: &'b (), c: &'c ()) {
    let x = match true {
        true => foo::<&'c ()>, //~ ERROR lifetime may not live long enough
        false => foo::<&'a ()>, //~ ERROR lifetime may not live long enough
    };

    x(a);
    x(b);
    x(c);
}

fn main() {}
