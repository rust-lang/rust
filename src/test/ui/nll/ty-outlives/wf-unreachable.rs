// Test that we check that user type annotations are well-formed, even in dead
// code.

#![feature(nll)]

fn uninit<'a>() {
    return;
    let x: &'static &'a ();                         //~ ERROR lifetime may not be long enough
}

fn var_type<'a>() {
    return;
    let x: &'static &'a () = &&();                  //~ ERROR lifetime may not be long enough
}

fn uninit_infer<'a>() {
    let x: &'static &'a _;                          //~ ERROR lifetime may not be long enough
    x = && ();
}

fn infer<'a>() {
    return;
    let x: &'static &'a _ = &&();                   //~ ERROR lifetime may not be long enough
}

fn uninit_no_var<'a>() {
    return;
    let _: &'static &'a ();                         //~ ERROR lifetime may not be long enough
}

fn no_var<'a>() {
    return;
    let _: &'static &'a () = &&();                  //~ ERROR lifetime may not be long enough
}

fn infer_no_var<'a>() {
    return;
    let _: &'static &'a _ = &&();                   //~ ERROR lifetime may not be long enough
}

trait X<'a, 'b> {}

struct C<'a, 'b, T: X<'a, 'b>>(T, &'a (), &'b ());

impl X<'_, '_> for i32 {}
impl<'a> X<'a, 'a> for () {}

// This type annotation is not well-formed because we substitute `()` for `_`.
fn required_substs<'a>() {
    return;
    let _: C<'static, 'a, _> = C((), &(), &());     //~ ERROR lifetime may not be long enough
}

fn main() {}
