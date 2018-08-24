// revisions: ast mir
//[mir]compile-flags: -Z borrowck=mir

#![feature(rustc_attrs)]

enum Sexpression {
    Num(()),
    Cons(&'static mut Sexpression)
}

fn causes_error_in_ast(mut l: &mut Sexpression) {
    loop { match l {
        &mut Sexpression::Num(ref mut n) => {},
        &mut Sexpression::Cons(ref mut expr) => { //[ast]~ ERROR [E0499]
            l = &mut **expr; //[ast]~ ERROR [E0506]
        }
    }}
}

#[rustc_error]
fn main() { //[mir]~ ERROR compilation successful
}
