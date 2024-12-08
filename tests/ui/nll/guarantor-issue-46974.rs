// Test that NLL analysis propagates lifetimes correctly through
// field accesses, Box accesses, etc.

fn foo(s: &mut (i32,)) -> i32 {
    let t = &mut *s; // this borrow should last for the entire function
    let x = &t.0;
    *s = (2,); //~ ERROR cannot assign to `*s`
    *x
}

fn bar(s: &Box<(i32,)>) -> &'static i32 {
    &s.0 //~ ERROR lifetime may not live long enough
}

fn main() {
    foo(&mut (0,));
    bar(&Box::new((1,)));
}
