#![feature(const_refs_to_static)]

static mut S_MUT: i32 = 0;

const C: &i32 = unsafe { &S_MUT };
//~^ERROR: constant refers to mutable data

fn main() {
    // This *must not build*, the constant we are matching against
    // could change its value!
    match &42 {
        C => {}, //~ERROR: could not evaluate constant pattern
        _ => {},
    }
}
