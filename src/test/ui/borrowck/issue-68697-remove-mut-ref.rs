// Regression test for #68697: Suggest removing `&mut x`
// when `x: &mut T` and `&mut T` is expected type

struct A;

fn bar(x: &mut A) {}

fn foo(x: &mut A) {
    bar(&mut x);
    //~^ ERROR: cannot borrow `x` as mutable
    //~| HELP: remove the unnecessary `&mut` here
    //~| SUGGESTION: x
}

fn main() {}
