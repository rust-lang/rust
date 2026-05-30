// Regression test for #87290.

//@ check-pass

#![crate_type = "lib"]

pub fn f<'a>(_: &'a i32, _: impl FnOnce(&mut &mut &'a i32)) {}

pub fn g<'a>(p: &'a i32) {
    f(p, |_: &mut &mut &'a i32| {})
}

pub fn h<'a>(p: &'a i32) {
    f(p, |x: &mut &mut &'a i32| {
        let _: &mut &mut &'a i32 = x;
    })
}
