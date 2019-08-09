// build-pass (FIXME(62277): could be check-pass?)

#![allow(dead_code, unused_variables)]

// Issue #21633:  reject duplicate loop labels in function bodies.
//
// Test rejection of lifetimes in *expressions* that shadow loop labels.

fn foo() {
    // Reusing lifetime `'a` in function item is okay.
    fn foo<'a>(x: &'a i8) -> i8 { *x }

    // So is reusing `'a` in struct item
    struct S1<'a> { x: &'a i8 } impl<'a> S1<'a> { fn m(&self) {} }
    // and a method item
    struct S2; impl S2 { fn m<'a>(&self) {} }

    let z = 3_i8;

    'a: loop {
        let b = Box::new(|x: &i8| *x) as Box<dyn for <'a> Fn(&'a i8) -> i8>;
        //~^ WARN lifetime name `'a` shadows a label name that is already in scope
        assert_eq!((*b)(&z), z);
        break 'a;
    }
}


pub fn main() {
    foo();
}
