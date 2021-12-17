#![feature(let_else)]

// Slightly different from explicit-mut-annotated -- this won't show an error until borrowck.
// Should it show a type error instead?

fn main() {
    let Some(n): &mut Option<i32> = &mut &Some(5i32) else {
        //~^ ERROR cannot borrow data in a `&` reference as mutable
        return
    };
    *n += 1;
    let _ = n;
}
