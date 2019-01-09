// Test that a (partially) mutably borrowed place can be matched on, so long as
// we don't have to read any values that are mutably borrowed to determine
// which arm to take.
//
// Test that we don't allow mutating the value being matched on in a way that
// changes which patterns it matches, until we have chosen an arm.

#![feature(nll)]

struct A(i32, i32);

fn struct_example(mut a: A) {
    let x = &mut a.0;
    match a { // OK, no access of borrowed data
        _ if false => (),
        A(_, r) => (),
    }
    x;
}

fn indirect_struct_example(mut b: &mut A) {
    let x = &mut b.0;
    match *b { // OK, no access of borrowed data
        _ if false => (),
        A(_, r) => (),
    }
    x;
}

fn underscore_example(mut c: i32) {
    let r = &mut c;
    match c { // OK, no access of borrowed data (or any data at all)
        _ if false => (),
        _ => (),
    }
    r;
}

enum E {
    V(i32, i32),
    W,
}

fn enum_example(mut e: E) {
    let x = match e {
        E::V(ref mut x, _) => x,
        E::W => panic!(),
    };
    match e { // Don't know that E uses a tag for its discriminant
        _ if false => (),
        E::V(_, r) => (), //~ ERROR
        E::W => (),
    }
    x;
}

fn indirect_enum_example(mut f: &mut E) {
    let x = match *f {
        E::V(ref mut x, _) => x,
        E::W => panic!(),
    };
    match f { // Don't know that E uses a tag for its discriminant
        _ if false => (),
        E::V(_, r) => (), //~ ERROR
        E::W => (),
    }
    x;
}

fn match_on_muatbly_borrowed_ref(mut p: &bool) {
    let r = &mut p;
    match *p { // OK, no access at all
        _ if false => (),
        _ => (),
    }
    r;
}

fn match_on_borrowed(mut t: bool) {
    let x = &mut t;
    match t {
        true => (), //~ ERROR
        false => (),
    }
    x;
}

enum Never {}

fn never_init() {
    let n: Never;
    match n {} //~ ERROR
}

fn main() {}
